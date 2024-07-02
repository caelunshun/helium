use crate::{cuda::error::CudaError, data_type::DataType};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, DriverError};
use half::{bf16, f16};
use std::sync::Arc;

/// Raw CUDA tensor.
pub struct RawTensor {
    shape: Vec<usize>,
    data: Data,
}

impl RawTensor {
    pub fn new(data: Data, shape: impl Into<Vec<usize>>) -> Self {
        Self {
            shape: shape.into(),
            data,
        }
    }

    pub fn dimension(&self) -> u32 {
        self.shape.len() as u32
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dim_at(&self, x: i32) -> usize {
        if x < 0 {
            self.shape[self.shape.len() - x.abs() as usize]
        } else {
            self.shape[x as usize]
        }
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().copied().sum()
    }

    pub fn data(&self) -> &Data {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut Data {
        &mut self.data
    }

    pub fn data_type(&self) -> DataType {
        self.data.data_type()
    }
}

pub enum Data {
    F32(CudaSlice<f32>),
    Bf16(CudaSlice<bf16>),
    F16(CudaSlice<f16>),
}

impl Data {
    pub unsafe fn alloc(
        device: &Arc<CudaDevice>,
        data_type: DataType,
        len: usize,
    ) -> Result<Self, DriverError> {
        match data_type {
            DataType::F32 => device.alloc(len).map(Self::F32),
            DataType::Bf16 => device.alloc(len).map(Self::Bf16),
            DataType::F16 => device.alloc(len).map(Self::F16),
        }
    }

    pub fn device_ptr(&self) -> *const u8 {
        match self {
            Data::F32(d) => *d.device_ptr() as usize as *const u8,
            Data::Bf16(d) => *d.device_ptr() as usize as *const u8,
            Data::F16(d) => *d.device_ptr() as usize as *const u8,
        }
    }

    pub fn device_ptr_mut(&mut self) -> *mut u8 {
        match self {
            Data::F32(d) => *d.device_ptr_mut() as usize as *mut u8,
            Data::Bf16(d) => *d.device_ptr_mut() as usize as *mut u8,
            Data::F16(d) => *d.device_ptr_mut() as usize as *mut u8,
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            Data::F32(_) => DataType::F32,
            Data::Bf16(_) => DataType::Bf16,
            Data::F16(_) => DataType::F16,
        }
    }
}
