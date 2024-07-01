use crate::{
    cuda::{
        error::CudaError,
        kernel::{CompiledKernel, Kernel, KernelParam},
    },
    opgraph::subgraph::OpSubgraph,
};
use ahash::AHashMap;
use cudarc::driver::CudaDevice;
use parking_lot::Mutex;
use std::{
    collections::hash_map::Entry,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

/// Shared state for caching CUDA values on a particular device.
///
/// Can be cloned like an `Arc`.
#[derive(Clone)]
pub struct Cuda {
    device: Arc<CudaDevice>,
    /// Maps subgraphs to compiled + loaded kernels.
    kernel_cache: Arc<Mutex<AHashMap<OpSubgraph, Arc<LoadedKernel>>>>,
    next_module_id: Arc<AtomicU64>,
}

impl Cuda {
    pub fn new(device_index: u32) -> Result<Self, CudaError> {
        let device = CudaDevice::new_with_stream(device_index as usize)?;
        Ok(Self {
            device,
            kernel_cache: Arc::new(Mutex::new(AHashMap::new())),
            next_module_id: Arc::new(AtomicU64::new(0)),
        })
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
                Ok(entry
                    .insert(Arc::new(LoadedKernel {
                        params: kernel.params().collect(),
                        module_name: module_id,
                        func_name: kernel.entrypoint_name(),
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
}
