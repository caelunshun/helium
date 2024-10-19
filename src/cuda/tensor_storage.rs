use crate::{
    cuda::{
        allocator::{Memory, StreamId},
        context::{CudaContext, CudaEvent, CudaStream},
        error::CudaError,
    },
    data_type::{DataType, DataVec},
};
use cudarc::{driver, driver::sys::CUdeviceptr};
use half::{bf16, f16};
use std::{
    mem,
    mem::MaybeUninit,
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
    memory: Arc<Memory>,
    data_type: DataType,
    id: TensorStorageId,

    /// If the tensor is currently being computed on the GPU,
    /// or a CPU->GPU transfer operation is in progress, this event
    /// will be signaled once said operation completes and the tensor
    /// is safe to use.
    ready_event: Arc<CudaEvent>,
}

impl TensorStorage {
    pub fn new(
        data_type: DataType,
        len: usize,
        cx: &CudaContext,
        allocation_stream: Option<StreamId>,
    ) -> Result<Self, CudaError> {
        const ALIGN: u64 = 64;

        // Align the size to a multiple of 32 bits
        let num_bytes = (len * data_type.size_in_bits() + 31) / 32 * 4;

        let memory = match allocation_stream {
            Some(allocation_stream) => {
                cx.allocator()
                    .allocate_in_stream(num_bytes as u64, ALIGN, allocation_stream)?
            }
            None => cx.allocator().allocate(num_bytes as u64, ALIGN)?,
        };
        Ok(Self {
            id: TensorStorageId::new(),
            memory: Arc::new(memory),
            data_type,
            ready_event: Arc::new(CudaEvent::new()?),
        })
    }

    pub fn id(&self) -> TensorStorageId {
        self.id
    }

    pub fn initialize_with_data(
        &self,
        data: &DataVec,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        assert_eq!(data.as_bytes().len(), self.memory.len() as usize);
        let bytes: &[u8] = data.as_bytes();
        unsafe {
            driver::result::memcpy_htod_async(self.device_ptr(), bytes, stream.raw() as _)?;
        }
        Ok(())
    }

    /// # Safety
    /// The returned `DataVec` must live until the operation completes
    /// in the stream.
    pub unsafe fn async_copy_to_host(&self, stream: &CudaStream) -> Result<DataVec, CudaError> {
        match self.data_type {
            DataType::F16 => {
                let mut data = vec![MaybeUninit::<f16>::uninit(); self.memory.len() as usize / 2];
                unsafe {
                    driver::result::memcpy_dtoh_async(&mut data, self.device_ptr(), stream.raw())?;
                    Ok(DataVec::F16(mem::transmute(data)))
                }
            }
            DataType::Bf16 => {
                let mut data = vec![MaybeUninit::<bf16>::uninit(); self.memory.len() as usize / 2];
                unsafe {
                    driver::result::memcpy_dtoh_async(&mut data, self.device_ptr(), stream.raw())?;
                    Ok(DataVec::Bf16(mem::transmute(data)))
                }
            }
            DataType::F32 => {
                let mut data = vec![MaybeUninit::<f32>::uninit(); self.memory.len() as usize / 4];
                unsafe {
                    driver::result::memcpy_dtoh_async(&mut data, self.device_ptr(), stream.raw())?;
                    Ok(DataVec::F32(mem::transmute(data)))
                }
            }
            DataType::U32 => {
                let mut data = vec![MaybeUninit::<u32>::uninit(); self.memory.len() as usize / 4];
                unsafe {
                    driver::result::memcpy_dtoh_async(&mut data, self.device_ptr(), stream.raw())?;
                    Ok(DataVec::U32(mem::transmute(data)))
                }
            }
            DataType::Bool => {
                let mut data = vec![MaybeUninit::<u32>::uninit(); self.memory.len() as usize / 4];
                unsafe {
                    driver::result::memcpy_dtoh_async(&mut data, self.device_ptr(), stream.raw())?;
                    Ok(DataVec::Bool(mem::transmute(data)))
                }
            }
        }
    }

    pub fn fill(&self, value: f32, stream: &CudaStream) -> Result<(), CudaError> {
        match self.data_type {
            DataType::F16 => unsafe {
                driver::sys::lib()
                    .cuMemsetD16Async(
                        self.device_ptr(),
                        f16::from_f32(value).to_bits(),
                        (self.memory.len() / mem::size_of::<f32>() as u64) as usize,
                        stream.raw(),
                    )
                    .result()?;
            },
            DataType::Bf16 => unsafe {
                driver::sys::lib()
                    .cuMemsetD16Async(
                        self.device_ptr(),
                        bf16::from_f32(value).to_bits(),
                        (self.memory.len() / mem::size_of::<f32>() as u64) as usize,
                        stream.raw(),
                    )
                    .result()?;
            },
            DataType::F32 | DataType::U32 => unsafe {
                driver::sys::lib()
                    .cuMemsetD32Async(
                        self.device_ptr(),
                        value.to_bits(),
                        (self.memory.len() / mem::size_of::<f32>() as u64) as usize,
                        stream.raw(),
                    )
                    .result()?;
            },
            DataType::Bool => {
                let b = value != 0.0;
                let mask = if b { 0xFF } else { 0 };
                unsafe {
                    driver::sys::lib()
                        .cuMemsetD8Async(
                            self.device_ptr(),
                            mask,
                            self.memory.len() as usize,
                            stream.raw(),
                        )
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
}
