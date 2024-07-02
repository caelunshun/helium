use crate::{
    cuda::{
        error::CudaError,
        kernel::{CompiledKernel, Kernel, KernelParam},
    },
    data_type::DataType,
    opgraph::{op::Op, subgraph::OpSubgraph, Intermediate, Node, NodeId},
};
use ahash::AHashMap;
use cudarc::{
    cublaslt,
    cublaslt::sys::cublasLtHandle_t,
    driver,
    driver::{sys::CUstream, CudaDevice},
};
use parking_lot::Mutex;
use std::{
    collections::hash_map::Entry,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use thread_local::ThreadLocal;

/// Shared state for caching CUDA values on a particular device.
///
/// Can be cloned like an `Arc`.
#[derive(Clone)]
pub struct CudaContext {
    device: Arc<CudaDevice>,
    stream_pool: Arc<ThreadLocal<Vec<CudaStream>>>,
    cublaslt_pool: Arc<ThreadLocal<CublasLtContext>>,
    /// Maps subgraphs to compiled + loaded kernels.
    kernel_cache: Arc<Mutex<AHashMap<OpSubgraph, Arc<LoadedKernel>>>>,
    next_module_id: Arc<AtomicU64>,
}

impl CudaContext {
    pub fn new(device_index: u32) -> Result<Self, CudaError> {
        let device = CudaDevice::new_with_stream(device_index as usize)?;
        Ok(Self {
            device,
            stream_pool: Arc::new(ThreadLocal::new()),
            cublaslt_pool: Arc::new(ThreadLocal::new()),
            kernel_cache: Arc::new(Mutex::new(AHashMap::new())),
            next_module_id: Arc::new(AtomicU64::new(0)),
        })
    }

    pub fn stream_pool(&self) -> Result<&[CudaStream], CudaError> {
        const STREAMS_PER_THREAD: usize = 4;
        self.stream_pool
            .get_or_try(|| {
                let mut streams = Vec::new();
                for _ in 0..STREAMS_PER_THREAD {
                    streams.push(CudaStream::new(&self.device)?);
                }
                Ok(streams)
            })
            .map(Vec::as_slice)
    }

    pub fn cublaslt_handle(&self) -> Result<&CublasLtContext, CudaError> {
        self.cublaslt_pool.get_or_try(|| CublasLtContext::new())
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    fn new_module_id(&self) -> String {
        let id = self.next_module_id.fetch_add(1, Ordering::Relaxed);
        format!("module{id}")
    }

    pub fn get_or_init_kernel(
        &self,
        subgraph: &OpSubgraph,
        generate_kernel: impl FnOnce(&OpSubgraph) -> Kernel,
    ) -> Result<Arc<LoadedKernel>, CudaError> {
        match self.kernel_cache.lock().entry(subgraph.clone()) {
            Entry::Occupied(entry) => Ok(entry.get().clone()),
            Entry::Vacant(entry) => {
                let kernel = generate_kernel(subgraph);
                let kernel = CompiledKernel::new(&kernel, &self.device)?;
                let module_id = self.new_module_id();
                self.device.load_ptx(
                    kernel.ptx().clone(),
                    &module_id,
                    &[kernel.entrypoint_name()],
                )?;

                let mut output_types = AHashMap::new();
                for param in kernel.params() {
                    if let KernelParam::Output(node) = param {
                        output_types
                            .insert(node, subgraph.graph().get(node).descriptor().data_type);
                    }
                }

                Ok(entry
                    .insert(Arc::new(LoadedKernel {
                        params: kernel.params().collect(),
                        module_name: module_id,
                        func_name: kernel.entrypoint_name(),
                        output_types,
                        reduction_output: subgraph.leafs().find(|id| {
                            matches!(
                                subgraph.graph().get(*id),
                                Node::Intermediate(Intermediate {
                                    op: Op::Reduce(_),
                                    ..
                                })
                            )
                        }),
                    }))
                    .clone())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadedKernel {
    pub params: Vec<KernelParam>,
    pub module_name: String,
    pub func_name: &'static str,
    pub output_types: AHashMap<NodeId, DataType>,
    pub reduction_output: Option<NodeId>,
}

impl LoadedKernel {
    pub fn first_tensor_input(&self) -> NodeId {
        self.params
            .iter()
            .find_map(|p| match p {
                KernelParam::Node(n) => Some(*n),
                _ => None,
            })
            .expect("kernel has no tensor inputs")
    }
}

/// Wrapper for a CUDA stream.
#[derive(Debug)]
pub struct CudaStream {
    stream: cudarc::driver::CudaStream,
}

impl CudaStream {
    pub fn new(device: &Arc<CudaDevice>) -> Result<Self, CudaError> {
        let stream = device.fork_default_stream()?;
        Ok(Self { stream })
    }

    pub fn raw(&self) -> CUstream {
        self.stream.stream
    }

    pub fn cudarc_stream(&self) -> &driver::CudaStream {
        &self.stream
    }
}

unsafe impl Send for CudaStream {}

pub struct CublasLtContext(cublasLtHandle_t);

impl CublasLtContext {
    pub fn new() -> Result<Self, CudaError> {
        let cx = cublaslt::result::create_handle()?;
        Ok(Self(cx))
    }

    pub fn raw(&self) -> cublasLtHandle_t {
        self.0
    }
}

unsafe impl Send for CublasLtContext {}
