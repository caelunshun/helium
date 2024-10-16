use crate::{
    cuda::{
        allocator::{Memory, StreamId},
        context::{CudaContext, CudaStream},
        error::CudaError,
    },
    data_type::{DataType, DataTypeConversion, DataVec},
};
use cudarc::{driver, driver::sys::CUdeviceptr};
use half::{bf16, f16};
use std::{mem, sync::Arc};

#[derive(Clone)]
pub struct TensorStorage {
    memory: Arc<Memory>,
    data_type: DataType,
}

impl TensorStorage {
    pub fn new(
        data_type: DataType,
        len: usize,
        cx: &CudaContext,
        allocation_stream: StreamId,
    ) -> Result<Self, CudaError> {
        const ALIGN: u64 = 64;
        let memory = cx.allocator().allocate_in_stream(
            len as u64 * data_type.size() as u64,
            ALIGN,
            allocation_stream,
        )?;
        Ok(Self {
            memory: Arc::new(memory),
            data_type,
        })
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
            DataType::F32 => unsafe {
                driver::sys::lib()
                    .cuMemsetD32Async(
                        self.device_ptr(),
                        value.to_bits(),
                        (self.memory.len() / mem::size_of::<f32>() as u64) as usize,
                        stream.raw(),
                    )
                    .result()?;
            },
        }
        Ok(())
    }

    pub fn device_ptr(&self) -> CUdeviceptr {
        self.memory.device_ptr()
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn to_vec<T: DataTypeConversion>(&self) -> Vec<T> {
        let mut bytes = vec![0u8; self.memory.len() as usize];
        unsafe {
            driver::result::memcpy_dtoh_sync(&mut bytes, self.device_ptr())
                .expect("failed to memcpy from device to host");
        }
        match self.data_type {
            DataType::F16 => bytemuck::cast_slice::<u8, f16>(&bytes)
                .iter()
                .copied()
                .map(f16::to_f32)
                .map(T::from_f32)
                .collect(),
            DataType::Bf16 => bytemuck::cast_slice::<u8, bf16>(&bytes)
                .iter()
                .copied()
                .map(bf16::to_f32)
                .map(T::from_f32)
                .collect(),
            DataType::F32 => bytemuck::cast_slice::<u8, f32>(&bytes)
                .iter()
                .copied()
                .map(T::from_f32)
                .collect(),
        }
    }
}
