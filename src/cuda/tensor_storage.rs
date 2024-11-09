use crate::{
    cuda::{
        allocator::{DeviceMemory, HostPinnedAllocator, StreamId},
        context::{CudaContext, CudaEvent, CudaStream},
        error::CudaError,
    },
    data_type::{DataSlice, DataType, DataVec},
};
use cudarc::{driver, driver::sys::CUdeviceptr};
use half::{bf16, f16};
use std::{
    alloc::Layout,
    mem, ptr, slice,
    sync::{atomic::AtomicU64, Arc},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorStorageId(u64);

impl TensorStorageId {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

#[derive(Clone)]
pub struct TensorStorage {
    memory: Arc<DeviceMemory>,
    num_bytes: usize,
    data_type: DataType,
    id: TensorStorageId,

    /// If the tensor is currently being computed on the GPU,
    /// or a CPU->GPU transfer operation is in progress, this event
    /// will be signaled once said operation completes and the tensor
    /// is safe to use.
    ready_event: Arc<CudaEvent>,
}

impl TensorStorage {
    const PAGE_LOCKED_MEM_ALIGNMENT: usize = 128;

    pub fn new(
        data_type: DataType,
        len: usize,
        cx: &CudaContext,
        allocation_stream: Option<StreamId>,
    ) -> Result<Self, CudaError> {
        const ALIGN: u64 = 64;

        let num_bytes = if data_type == DataType::Bool {
            // Align the size to a multiple of 32 bits
            (len * data_type.size_in_bits() + 31) / 32 * 4
        } else {
            len * data_type.size_in_bits() / 8
        };

        let memory = match allocation_stream {
            Some(allocation_stream) => {
                cx.allocator()
                    .allocate_in_stream(num_bytes as u64, ALIGN, allocation_stream)?
            }
            None => cx.allocator().allocate(num_bytes as u64, ALIGN)?,
        };
        Ok(Self {
            id: TensorStorageId::new(),
            num_bytes,
            memory: Arc::new(memory),
            data_type,
            ready_event: Arc::new(CudaEvent::new()?),
        })
    }

    pub fn id(&self) -> TensorStorageId {
        self.id
    }

    #[profiling::function]
    pub fn initialize_with_data(
        &self,
        data: DataSlice,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        assert_eq!(data.as_bytes().len(), self.num_bytes);
        let bytes: &[u8] = data.as_bytes();

        // Copy to page-locked memory for true asynchronous transfer.
        let page_locked = HostPinnedAllocator::global().alloc(
            Layout::from_size_align(bytes.len(), Self::PAGE_LOCKED_MEM_ALIGNMENT).unwrap(),
        )?;

        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), page_locked.as_ptr(), bytes.len());

            let page_locked_slice = slice::from_raw_parts(page_locked.as_ptr(), bytes.len());

            driver::result::memcpy_htod_async(
                self.device_ptr(),
                page_locked_slice,
                stream.raw() as _,
            )?;

            stream.insert_host_callback(move || {
                drop(page_locked);
            })?;
        }
        Ok(())
    }

    pub fn async_copy_to_host(
        &self,
        stream: &CudaStream,
        callback: impl FnOnce(DataVec) + Send + 'static,
    ) -> Result<(), CudaError> {
        // First copy to page-locked memory, then to normal Vec memory.
        let num_bytes = self.num_bytes;
        let page_locked = HostPinnedAllocator::global()
            .alloc(Layout::from_size_align(num_bytes, Self::PAGE_LOCKED_MEM_ALIGNMENT).unwrap())?;
        unsafe {
            let page_locked_slice = slice::from_raw_parts_mut(page_locked.as_ptr(), num_bytes);
            driver::result::memcpy_dtoh_async(page_locked_slice, self.device_ptr(), stream.raw())?;
        }

        let data_type = self.data_type;

        // Tensor memory needs to stay alive until transfer completes.
        let device_memory_handle = self.memory.clone();

        stream.insert_host_callback(move || {
            let bytes = unsafe { slice::from_raw_parts(page_locked.as_ptr(), num_bytes) };

            let data_vec = match data_type {
                DataType::F16 => {
                    let mut data = vec![f16::ZERO; num_bytes / 2];
                    data.copy_from_slice(bytemuck::cast_slice(bytes));
                    DataVec::F16(data)
                }
                DataType::Bf16 => {
                    let mut data = vec![bf16::ZERO; num_bytes / 2];
                    data.copy_from_slice(bytemuck::cast_slice(bytes));
                    DataVec::Bf16(data)
                }
                DataType::F32 => {
                    let mut data = vec![0.0f32; num_bytes / 4];
                    data.copy_from_slice(bytemuck::cast_slice(bytes));
                    DataVec::F32(data)
                }
                DataType::U32 => {
                    let mut data = vec![0u32; num_bytes / 4];
                    data.copy_from_slice(bytemuck::cast_slice(bytes));
                    DataVec::U32(data)
                }
                DataType::Bool => {
                    let mut data = vec![0u32; num_bytes / 4];
                    data.copy_from_slice(bytemuck::cast_slice(bytes));
                    DataVec::Bool(data)
                }
            };

            drop(page_locked);
            drop(device_memory_handle);

            callback(data_vec);
        })?;

        Ok(())
    }

    pub fn fill(&self, value: f32, stream: &CudaStream) -> Result<(), CudaError> {
        match self.data_type {
            DataType::F16 => unsafe {
                driver::sys::lib()
                    .cuMemsetD16Async(
                        self.device_ptr(),
                        f16::from_f32(value).to_bits(),
                        self.num_bytes / mem::size_of::<f32>(),
                        stream.raw(),
                    )
                    .result()?;
            },
            DataType::Bf16 => unsafe {
                driver::sys::lib()
                    .cuMemsetD16Async(
                        self.device_ptr(),
                        bf16::from_f32(value).to_bits(),
                        self.num_bytes / mem::size_of::<f32>(),
                        stream.raw(),
                    )
                    .result()?;
            },
            DataType::F32 | DataType::U32 => unsafe {
                driver::sys::lib()
                    .cuMemsetD32Async(
                        self.device_ptr(),
                        value.to_bits(),
                        self.num_bytes / mem::size_of::<f32>(),
                        stream.raw(),
                    )
                    .result()?;
            },
            DataType::Bool => {
                let b = value != 0.0;
                let mask = if b { 0xFF } else { 0 };
                unsafe {
                    driver::sys::lib()
                        .cuMemsetD8Async(self.device_ptr(), mask, self.num_bytes, stream.raw())
                        .result()?;
                }
            }
        }
        Ok(())
    }

    pub fn ready_event(&self) -> &Arc<CudaEvent> {
        &self.ready_event
    }

    pub fn device_ptr(&self) -> CUdeviceptr {
        self.memory.device_ptr()
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn mark_in_use_by_stream(&self, stream: StreamId) {
        self.memory.mark_in_use_by_stream(stream);
    }
}
