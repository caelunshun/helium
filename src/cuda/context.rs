use crate::cuda::{
    allocator::{DeviceAllocator, StreamId},
    cudnn::CudnnContext,
    error::CudaError,
};
use cudarc::{
    driver,
    driver::{
        sys::{
            CUdevice_attribute_enum::{
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            },
            CUevent, CUevent_flags_enum, CUevent_wait_flags_enum, CUstream,
        },
        CudaDevice,
    },
    nvrtc::Ptx,
};
use parking_lot::{Mutex, MutexGuard, RwLock};
use std::{
    cell::Cell,
    ffi::c_void,
    iter,
    sync::{Arc, OnceLock},
    time::Duration,
};
use thread_local::ThreadLocal;

/// Shared state for caching CUDA values on a particular device.
pub struct CudaContext {
    device: Arc<CudaDevice>,
    allocator: Mutex<DeviceAllocator>,
    stream_pool: ThreadLocal<Vec<CudaStream>>,
    cudnn_pool: ThreadLocal<CudnnContext>,
    sm_version: u32,
    loaded_kernels: Mutex<Vec<Arc<Ptx>>>,
    previous_alloc_stream: ThreadLocal<Cell<Option<StreamId>>>,

    /// Streams for memory transfer operations.
    dtoh_stream: CudaStream,
    htod_stream: CudaStream,
}

impl CudaContext {
    pub fn global(device_index: u32) -> Result<&'static Self, CudaError> {
        static CONTEXTS: OnceLock<RwLock<Vec<Option<&'static CudaContext>>>> = OnceLock::new();
        let lock = CONTEXTS.get_or_init(Default::default);

        // Optimistic check with read-only lock
        let guard = lock.read();
        if let Some(Some(cx)) = guard.get(device_index as usize) {
            cx.device.bind_to_thread()?;
            return Ok(*cx);
        }

        drop(guard);
        let mut guard = lock.write();
        if let Some(Some(cx)) = guard.get(device_index as usize) {
            cx.device.bind_to_thread()?;
            return Ok(*cx);
        }

        let needed_space = device_index as usize + 1 - guard.len();
        guard.extend(iter::repeat(None).take(needed_space));
        Ok(guard[device_index as usize].insert(Box::leak(Box::new(Self::new(device_index)?))))
    }

    pub fn new(device_index: u32) -> Result<Self, CudaError> {
        let device = CudaDevice::new_with_stream(device_index as usize)?;

        let allocator = unsafe { DeviceAllocator::new(*device.cu_primary_ctx()) };

        let compute_major = device.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let compute_minor = device.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;

        Ok(Self {
            dtoh_stream: CudaStream::new(&device)?,
            htod_stream: CudaStream::new(&device)?,
            device,
            allocator: Mutex::new(allocator),
            stream_pool: ThreadLocal::new(),
            cudnn_pool: ThreadLocal::new(),
            sm_version: (compute_major * 10 + compute_minor) as u32,
            loaded_kernels: Mutex::new(Vec::new()),
            previous_alloc_stream: ThreadLocal::new(),
        })
    }

    pub fn stream_pool(&self) -> Result<&[CudaStream], CudaError> {
        const STREAMS_PER_THREAD: usize = 2;
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

    pub fn cudnn_handle(&self) -> &CudnnContext {
        self.cudnn_pool.get_or(|| {
            self.device.bind_to_thread().unwrap();
            CudnnContext::new().expect("failed to init cuDNN")
        })
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn allocator(&self) -> MutexGuard<DeviceAllocator> {
        self.allocator.lock()
    }

    pub fn sm_version(&self) -> u32 {
        self.sm_version
    }

    pub fn load_kernel_if_needed(
        &self,
        kernel: &Arc<Ptx>,
        module_name: &str,
        func_name: &str,
    ) -> Result<(), CudaError> {
        let mut kernels = self.loaded_kernels.lock();
        if kernels.iter().any(|kernel2| Arc::ptr_eq(kernel, kernel2)) {
            return Ok(());
        }
        self.device.load_ptx(
            (**kernel).clone(),
            module_name,
            &[func_name.to_owned().leak()],
        )?;
        kernels.push(kernel.clone());
        Ok(())
    }

    pub fn previous_alloc_stream_for_thread(&self) -> Option<StreamId> {
        self.previous_alloc_stream.get_or_default().get()
    }

    pub fn set_previous_alloc_stream_for_thread(&self, stream: StreamId) {
        self.previous_alloc_stream
            .get_or_default()
            .set(Some(stream));
    }

    pub fn dtoh_stream(&self) -> &CudaStream {
        &self.dtoh_stream
    }

    pub fn htod_stream(&self) -> &CudaStream {
        &self.htod_stream
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

    pub fn insert_host_callback<F: FnOnce() + Send + 'static>(
        &self,
        callback: F,
    ) -> Result<(), CudaError> {
        unsafe extern "C" fn host_callback<F: FnOnce() + Send + 'static>(user_data: *mut c_void) {
            let f = Box::from_raw(user_data.cast::<F>());
            f();
        }

        let callback = Box::into_raw(Box::new(callback)).cast::<c_void>();
        unsafe {
            driver::sys::lib()
                .cuLaunchHostFunc(self.raw(), Some(host_callback::<F>), callback)
                .result()?;
        }
        Ok(())
    }

    pub fn raw(&self) -> CUstream {
        self.stream.stream
    }

    pub fn cudarc_stream(&self) -> &driver::CudaStream {
        &self.stream
    }
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

/// Wrapper for CUevent for synchronization on streams.
pub struct CudaEvent {
    raw: CUevent,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    pub fn new() -> Result<Self, CudaError> {
        Ok(Self {
            raw: driver::result::event::create(CUevent_flags_enum::CU_EVENT_BLOCKING_SYNC)?,
        })
    }

    pub fn record(&self, stream: &CudaStream) -> Result<(), CudaError> {
        unsafe {
            driver::result::event::record(self.raw, stream.raw())?;
        }
        Ok(())
    }

    pub fn wait(&self, stream: &CudaStream) -> Result<(), CudaError> {
        unsafe {
            driver::result::stream::wait_event(
                stream.raw(),
                self.raw,
                CUevent_wait_flags_enum::CU_EVENT_WAIT_DEFAULT,
            )?;
        }
        Ok(())
    }

    pub fn sync(&self) -> Result<(), CudaError> {
        unsafe {
            driver::sys::lib().cuEventSynchronize(self.raw).result()?;
        }
        Ok(())
    }

    pub fn measure_time_elapsed(&self, start: &CudaEvent) -> Result<Duration, CudaError> {
        self.sync()?;
        let mut millis = 0.0f32;
        unsafe {
            driver::sys::lib()
                .cuEventElapsedTime(&mut millis, start.raw, self.raw)
                .result()?;
        }
        Ok(Duration::from_secs_f32(millis / 1000.0))
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            driver::result::event::destroy(self.raw).expect("failed to destroy event");
        }
    }
}
