use crate::{
    cuda::{
        context::{CudaContext, CudaStream},
        error::CudaError,
        tensor_storage::TensorStorage,
    },
    opgraph::NodeId,
    DataType,
};
use ahash::AHashMap;
use bumpalo::Bump;
use cudarc::{
    driver::{LaunchAsync, LaunchConfig},
    nvrtc,
    nvrtc::Ptx,
};
use indoc::{formatdoc, indoc};
use parking_lot::Mutex;
use std::{
    cell::Cell,
    collections::hash_map::Entry,
    ffi::c_void,
    sync::{Arc, OnceLock},
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
            items: vec![indoc! {"
                    typedef unsigned int uint32_t;
        
                    struct alignas(8) __nv_bfloat164 {
                        __nv_bfloat16 x;
                        __nv_bfloat16 y;
                        __nv_bfloat16 z;
                        __nv_bfloat16 w;
                    };

                    struct alignas(8) half4 {
                        half x;
                        half y;
                        half z;
                        half w;
                    };
                "}
            .to_owned()],
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

    #[expect(unused)]
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
        formatdoc! {"
            {includes}
            {items}

            extern \"C\" __global__ void {kernel_name}({params}) {{
                {statements}
            }}
        "}
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

        static CACHE: OnceLock<Mutex<AHashMap<CacheKey, Arc<Ptx>>>> = OnceLock::new();

        let source = self.build_source(kernel_name);
        let key = CacheKey {
            source,
            sm_version: target_device.sm_version(),
        };
        match CACHE.get_or_init(Default::default).lock().entry(key) {
            Entry::Occupied(entry) => Ok(entry.get().clone()),
            Entry::Vacant(entry) => {
                let arch = format!("sm_{}", target_device.sm_version()).leak();
                let ptx = nvrtc::safe::compile_ptx_with_opts(
                    &entry.key().source,
                    nvrtc::safe::CompileOptions {
                        ftz: None,
                        prec_sqrt: None,
                        prec_div: None,
                        fmad: None,
                        options: vec![],
                        use_fast_math: None,
                        maxrregcount: None,
                        include_paths: vec!["/usr/local/cuda/include".to_owned()], // TODO Linux only
                        arch: Some(arch),
                    },
                )?;
                let ptx = Arc::new(ptx);
                entry.insert(ptx.clone());
                Ok(ptx)
            }
        }
    }

    pub fn build(
        &self,
        kernel_name: &str,
        target_device: &CudaContext,
    ) -> Result<JitKernel, CudaError> {
        let ptx = self.build_ptx(kernel_name, target_device)?;
        Ok(JitKernel {
            module_name: JitKernel::make_module_name(kernel_name, &ptx),
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
        }
    }

    #[expect(unused)]
    pub fn cpp_vector4_type(data_type: DataType) -> &'static str {
        match data_type {
            DataType::F16 => "half4",
            DataType::Bf16 => "__nv_bfloat164",
            DataType::F32 => "float4",
        }
    }

    #[expect(unused)]
    pub fn vector_component(index: usize) -> &'static str {
        match index {
            0 => "x",
            1 => "y",
            2 => "z",
            3 => "w",
            _ => panic!("vector component {index} out of bounds"),
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
    module_name: String,
}

impl JitKernel {
    fn make_module_name(kernel_name: &str, ptx: &Arc<Ptx>) -> String {
        let addr = Arc::as_ptr(ptx) as usize;
        format!("{kernel_name}_{addr:x}")
    }

    fn load_on_device(&self, device: &CudaContext) -> Result<(), CudaError> {
        device.load_kernel_if_needed(&self.ptx, &self.module_name, &self.kernel_name)
    }

    pub fn execute<'a>(
        &self,
        get_tensor_storage: impl Fn(NodeId) -> &'a TensorStorage,
        stream: &CudaStream,
        cx: &CudaContext,
        grid_size: u32,
        block_size: u32,
    ) -> Result<(), CudaError> {
        self.load_on_device(cx)?;

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
            cx.device()
                .get_func(&self.module_name, &self.kernel_name)
                .expect("loaded on device")
                .launch_on_stream(
                    stream.cudarc_stream(),
                    LaunchConfig {
                        grid_dim: (grid_size, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut params,
                )?;
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
