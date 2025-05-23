use crate::{
    DataType,
    cache::Cache,
    cuda::{
        context::{CudaContext, CudaStream},
        error::CudaError,
        tensor_storage::TensorStorage,
    },
    opgraph::NodeId,
};
use bumpalo::Bump;
use cudarc::{driver::sys::cuLaunchKernel, nvrtc, nvrtc::Ptx};
use indoc::formatdoc;
use std::{
    cell::Cell,
    ffi::c_void,
    ptr,
    sync::{Arc, atomic::AtomicU64},
};

/// Utility to build JIT CUDA kernels from C++ source.
#[derive(Debug)]
pub struct KernelBuilder {
    include_headers: Vec<String>,
    items: Vec<String>,
    params: Vec<KernelParam>,
    param_declarations: Vec<String>,
    statements: Vec<String>,
    next_ident: Cell<u64>,
}

impl Default for KernelBuilder {
    fn default() -> Self {
        Self {
            include_headers: vec![
                "cuda_bf16.h".to_owned(),
                "cuda_fp16.h".to_owned(),
                "vector_types.h".to_owned(),
            ],

            items: vec!["typedef unsigned int uint32_t;".to_owned()],
            params: vec![],
            param_declarations: vec![],
            statements: vec![],
            next_ident: Cell::new(0),
        }
    }
}

impl KernelBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn include(&mut self, header: &str) -> &mut Self {
        if !self.include_headers.iter().any(|h| h == header) {
            self.include_headers.push(header.to_owned());
        }
        self
    }

    pub fn new_ident(&self) -> Ident {
        let ident = format!("ident{}", self.next_ident.get());
        self.next_ident.set(self.next_ident.get() + 1);
        ident
    }

    pub fn param(&mut self, param: KernelParam, data_type: DataType) -> Ident {
        let ident = self.new_ident();
        self.params.push(param);
        self.param_declarations
            .push(format!("{} *{ident}", Self::cpp_data_type(data_type)));
        ident
    }

    pub fn statement(&mut self, statement: impl AsRef<str>) -> &mut Self {
        self.statements.push(statement.as_ref().to_owned());
        self
    }

    pub fn item(&mut self, item: impl AsRef<str>) -> &mut Self {
        self.items.push(item.as_ref().to_owned());
        self
    }

    pub fn build_source(&self, kernel_name: &str) -> String {
        let Self {
            statements,
            items,
            include_headers,
            param_declarations,
            ..
        } = self;
        let includes = include_headers
            .iter()
            .map(|h| format!("#include <{h}>\n"))
            .collect::<String>();
        let items = items.join("\n");
        let params = param_declarations.join(", ");
        let statements = statements.join("\n");
        let k = formatdoc! {"
            {includes}
            {items}

            extern \"C\" __global__ void {kernel_name}({params}) {{
                {statements}
            }}
        "};
        /*
        static ID: AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        std::fs::write(
            format!("kernel/{}.cu", ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)),
            k.as_bytes(),
        )
        .unwrap();
        */
        k
    }

    /// Compiles a PTX kernel targeting the given device.
    /// Uses a global cache to avoid compiling
    /// the same source twice.
    fn build_ptx(
        &self,
        kernel_name: &str,
        target_device: &CudaContext,
    ) -> Result<Arc<Ptx>, CudaError> {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        struct CacheKey {
            source: String,
            sm_version: u32,
        }

        static CACHE: Cache<CacheKey, Arc<Ptx>> = Cache::with_capacity(4096);

        let source = self.build_source(kernel_name);
        let key = CacheKey {
            source,
            sm_version: target_device.sm_version(),
        };

        static ID: AtomicU64 = AtomicU64::new(0);
        let id = ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let name = format!("jit_{kernel_name}_{id}.cu");
        #[cfg(feature = "cuda-debug")]
        {
            if !std::path::Path::new("kernel").exists() {
                std::fs::create_dir_all("kernel").ok();
            }
            std::fs::write(format!("kernel/{name}"), key.source.as_bytes()).unwrap();
        }

        Ok(CACHE.get_or_insert(&key, || {
            let arch = format!("sm_{}", target_device.sm_version()).leak();
            let ptx = nvrtc::safe::compile_ptx_with_opts(
                &key.source,
                nvrtc::safe::CompileOptions {
                    name: Some(name),
                    ftz: None,
                    prec_sqrt: None,
                    prec_div: None,
                    fmad: None,
                    options: vec!["-dopt=on".into(), "-lineinfo".into()],
                    use_fast_math: None,
                    maxrregcount: None,
                    include_paths: vec!["/usr/local/cuda/include".to_owned()], // TODO Linux only
                    arch: Some(arch),
                },
            )
            .unwrap_or_else(|e| {
                panic!(
                    "failed to compile kernel to PTX ({e}). source code: {}",
                    key.source
                )
            });
            Arc::new(ptx)
        }))
    }

    #[profiling::function]
    pub fn build(
        &self,
        kernel_name: &str,
        target_device: &CudaContext,
    ) -> Result<JitKernel, CudaError> {
        let ptx = self.build_ptx(kernel_name, target_device)?;
        Ok(JitKernel {
            ptx,
            params: self.params.clone(),
            kernel_name: kernel_name.to_owned(),
        })
    }

    pub fn cpp_data_type(data_type: DataType) -> &'static str {
        match data_type {
            DataType::F16 => "__half",
            DataType::Bf16 => "__nv_bfloat16",
            DataType::F32 => "float",
            DataType::U32 => "uint32_t",
            // Note: bitset packed
            DataType::Bool => "uint32_t",
        }
    }
}

/// Parameter passed to a kernel.
#[derive(Debug, Clone)]
pub enum KernelParam {
    /// Pointer to an input tensor of the subgraph.
    Input(NodeId),
    /// Pointer to an output tensor of the subgraph.
    Output(NodeId),
}

/// C++ identifier.
pub type Ident = String;

/// JIT'd kernel.
pub struct JitKernel {
    ptx: Arc<Ptx>,
    params: Vec<KernelParam>,
    kernel_name: String,
}

impl JitKernel {
    pub fn execute<'a>(
        &self,
        get_tensor_storage: impl Fn(NodeId) -> &'a TensorStorage,
        stream: &CudaStream,
        cx: &CudaContext,
        grid_size: u32,
        block_size: u32,
    ) -> Result<(), CudaError> {
        self.execute2d(
            get_tensor_storage,
            stream,
            cx,
            [grid_size, 1],
            [block_size, 1],
        )
    }

    pub fn execute2d<'a>(
        &self,
        get_tensor_storage: impl Fn(NodeId) -> &'a TensorStorage,
        stream: &CudaStream,
        cx: &CudaContext,
        grid_size: [u32; 2],
        block_size: [u32; 2],
    ) -> Result<(), CudaError> {
        let module = cx.load_module(&self.ptx)?;
        let function = module.get_function(&self.kernel_name)?;

        let bump = Bump::new();
        let mut params = self
            .params
            .iter()
            .map(|p| match p {
                KernelParam::Input(node) | KernelParam::Output(node) => {
                    let tensor_storage = get_tensor_storage(*node);
                    bump.alloc(tensor_storage.device_ptr()) as *mut _ as *mut c_void
                }
            })
            .collect::<Vec<_>>();

        unsafe {
            cuLaunchKernel(
                function,
                grid_size[0],
                grid_size[1],
                1,
                block_size[0],
                block_size[1],
                1,
                0,
                stream.raw(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            )
            .result()?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use slotmap::Key;

    #[test]
    fn compile_simple_kernel() {
        let context = CudaContext::global(0).unwrap();
        let mut kernel = KernelBuilder::new();
        let a = kernel.param(KernelParam::Input(NodeId::null()), DataType::Bf16);
        let b = kernel.param(KernelParam::Output(NodeId::null()), DataType::F16);
        kernel
            .statement("uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;")
            .statement(format!(
                "{a}[i] = static_cast<__nv_bfloat16>(static_cast<float>({b}[i]));"
            ));

        insta::assert_snapshot!(kernel.build_source("copy_kernel"));

        kernel.build("copy_kernel", context).unwrap();
    }
}
