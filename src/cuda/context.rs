use crate::cuda::error::CudaError;
use cudarc::{
    driver,
    driver::{
        sys::{CUmemPool_attribute_enum, CUstream},
        CudaDevice,
    },
};
use parking_lot::RwLock;
use std::{
    iter, ptr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, OnceLock,
    },
};
use thread_local::ThreadLocal;

/// Shared state for caching CUDA values on a particular device.
pub struct CudaContext {
    device: Arc<CudaDevice>,
    stream_pool: ThreadLocal<Vec<CudaStream>>,
    next_module_id: AtomicU64,
}

impl CudaContext {
    pub fn global(device_index: u32) -> Result<&'static Self, CudaError> {
        static CONTEXTS: OnceLock<RwLock<Vec<Option<&'static CudaContext>>>> = OnceLock::new();
        let lock = CONTEXTS.get_or_init(Default::default);

        // Optimistic check with read-only lock
        let guard = lock.read();
        if let Some(Some(cx)) = guard.get(device_index as usize) {
            return Ok(*cx);
        }

        drop(guard);
        let mut guard = lock.write();
        if let Some(Some(cx)) = guard.get(device_index as usize) {
            return Ok(*cx);
        }

        let needed_space = device_index as usize + 1 - guard.len();
        guard.extend(iter::repeat(None).take(needed_space));
        Ok(guard[device_index as usize].insert(Box::leak(Box::new(Self::new(device_index)?))))
    }

    pub fn new(device_index: u32) -> Result<Self, CudaError> {
        let device = CudaDevice::new_with_stream(device_index as usize)?;

        unsafe {
            let mut mem_pool = ptr::null_mut();

            driver::sys::lib().cuDeviceGetDefaultMemPool(&mut mem_pool, *device.cu_device());
            let mut release_threshold = 1024u64 * 1024 * 1024;
            driver::sys::lib().cuMemPoolSetAttribute(
                mem_pool,
                CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &mut release_threshold as *mut _ as *mut _,
            );
        }

        Ok(Self {
            device,
            stream_pool: ThreadLocal::new(),
            next_module_id: AtomicU64::new(0),
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

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    fn new_module_id(&self) -> String {
        let id = self.next_module_id.fetch_add(1, Ordering::Relaxed);
        format!("module{id}")
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
