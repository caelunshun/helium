use crate::cuda::{
    allocator::{DeviceAllocator, StreamId},
    cudnn::CudnnContext,
    error::CudaError,
};
use cudarc::{
    driver,
    driver::sys::{
        CUctx_flags::CU_CTX_MAP_HOST,
        CUdevice_attribute::{
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        },
        CUstream_flags::CU_STREAM_NON_BLOCKING,
        *,
    },
    nvrtc::Ptx,
};
use parking_lot::{Mutex, MutexGuard, RwLock};
use std::{
    cell::Cell,
    ffi::{CString, c_void},
    iter, ptr,
    str::FromStr,
    sync::{Arc, OnceLock},
    time::Duration,
};
use thread_local::ThreadLocal;

/// Shared state for caching CUDA values on a particular device.
pub struct CudaContext {
    ctx: CUcontext,
    device: CUdevice,
    allocator: Mutex<DeviceAllocator>,
    stream_pool: ThreadLocal<Vec<CudaStream>>,
    cudnn_pool: ThreadLocal<CudnnContext>,
    sm_version: u32,
    loaded_modules: Mutex<Vec<(Arc<Ptx>, Arc<CudaModule>)>>,
    previous_alloc_stream: ThreadLocal<Cell<Option<StreamId>>>,

    /// Streams for memory transfer operations.
    dtoh_stream: CudaStream,
    htod_stream: CudaStream,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

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
        driver::result::init()?;
        unsafe {
            let mut device: CUdevice = 0;
            cuDeviceGet(&mut device, device_index as i32).result()?;

            let mut ctx: CUcontext = ptr::null_mut();
            cuCtxCreate_v2(&mut ctx, CU_CTX_MAP_HOST as u32, device).result()?;

            let allocator = DeviceAllocator::new(ctx);

            let mut compute_major = 0;
            cuDeviceGetAttribute(
                &mut compute_major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device,
            )
            .result()?;
            let mut compute_minor = 0;
            cuDeviceGetAttribute(
                &mut compute_minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device,
            )
            .result()?;

            Ok(Self {
                ctx,
                device,
                dtoh_stream: CudaStream::new(ctx)?,
                htod_stream: CudaStream::new(ctx)?,
                allocator: Mutex::new(allocator),
                stream_pool: ThreadLocal::new(),
                cudnn_pool: ThreadLocal::new(),
                sm_version: (compute_major * 10 + compute_minor) as u32,
                loaded_modules: Mutex::new(Vec::new()),
                previous_alloc_stream: ThreadLocal::new(),
            })
        }
    }

    pub fn stream_pool(&self) -> Result<&[CudaStream], CudaError> {
        const STREAMS_PER_THREAD: usize = 2;
        self.stream_pool
            .get_or_try(|| {
                let mut streams = Vec::new();
                for _ in 0..STREAMS_PER_THREAD {
                    unsafe {
                        streams.push(CudaStream::new(self.ctx)?);
                    }
                }
                Ok(streams)
            })
            .map(Vec::as_slice)
    }

    pub fn cudnn_handle(&self) -> &CudnnContext {
        self.cudnn_pool.get_or(|| {
            self.bind_to_thread().unwrap();
            CudnnContext::new().expect("failed to init cuDNN")
        })
    }

    pub fn device(&self) -> CUdevice {
        self.device
    }

    pub fn raw_context(&self) -> CUcontext {
        self.ctx
    }

    pub fn allocator(&self) -> MutexGuard<DeviceAllocator> {
        self.allocator.lock()
    }

    pub fn sm_version(&self) -> u32 {
        self.sm_version
    }

    pub fn load_module(&self, kernel: &Arc<Ptx>) -> Result<Arc<CudaModule>, CudaError> {
        let mut modules = self.loaded_modules.lock();
        if let Some((_, module)) = modules
            .iter()
            .find(|(kernel2, _)| Arc::ptr_eq(kernel, kernel2))
        {
            return Ok(module.clone());
        }
        self.bind_to_thread()?;
        unsafe {
            let mut module: CUmodule = ptr::null_mut();
            cuModuleLoadData(
                &mut module,
                CString::from_str(&kernel.to_src()).unwrap().as_ptr().cast(),
            )
            .result()?;
            let module = Arc::new(CudaModule { raw: module });
            modules.push((kernel.clone(), module.clone()));
            Ok(module)
        }
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

    fn bind_to_thread(&self) -> Result<(), CudaError> {
        unsafe {
            cuCtxSetCurrent(self.ctx).result()?;
        }
        Ok(())
    }
}

/// Wrapper for a CUDA stream.
#[derive(Debug)]
pub struct CudaStream {
    stream: CUstream,
}

impl CudaStream {
    pub unsafe fn new(ctx: CUcontext) -> Result<Self, CudaError> {
        let mut stream: CUstream = ptr::null_mut();
        unsafe {
            cuCtxSetCurrent(ctx).result()?;
            cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING as u32).result()?;
        }
        Ok(Self { stream })
    }

    pub fn insert_host_callback<F: FnOnce() + Send + 'static>(
        &self,
        callback: F,
    ) -> Result<(), CudaError> {
        unsafe extern "C" fn host_callback<F: FnOnce() + Send + 'static>(user_data: *mut c_void) {
            let f = unsafe { Box::from_raw(user_data.cast::<F>()) };
            f();
        }

        let callback = Box::into_raw(Box::new(callback)).cast::<c_void>();
        unsafe {
            cuLaunchHostFunc(self.raw(), Some(host_callback::<F>), callback).result()?;
        }
        Ok(())
    }

    pub fn raw(&self) -> CUstream {
        self.stream
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            cuStreamDestroy_v2(self.stream).result().unwrap();
        }
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
    pub fn new(ctx: &CudaContext) -> Result<Self, CudaError> {
        ctx.bind_to_thread()?;
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
            cuEventSynchronize(self.raw).result()?;
        }
        Ok(())
    }

    pub fn measure_time_elapsed(&self, start: &CudaEvent) -> Result<Duration, CudaError> {
        self.sync()?;
        let mut millis = 0.0f32;
        unsafe {
            cuEventElapsedTime(&mut millis, start.raw, self.raw).result()?;
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

pub struct CudaModule {
    raw: CUmodule,
}

impl CudaModule {
    pub fn get_function(&self, name: &str) -> Result<CUfunction, CudaError> {
        unsafe {
            let mut func: CUfunction = ptr::null_mut();
            cuModuleGetFunction(
                &mut func,
                self.raw,
                CString::from_str(name).unwrap().as_ptr().cast(),
            )
            .result()?;
            Ok(func)
        }
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        unsafe {
            cuModuleUnload(self.raw).result().unwrap();
        }
    }
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}
