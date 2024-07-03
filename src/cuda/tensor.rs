use crate::{
    cuda::{context::CudaStream, error::CudaError},
    data_type::DataType,
};
use cudarc::{
    driver,
    driver::{sys::CUdeviceptr, CudaDevice},
};
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

    pub fn into_data(self) -> Data {
        self.data
    }
}

pub struct Data {
    device: Arc<CudaDevice>,
    ptr: CUdeviceptr,
    typ: DataType,
}

unsafe impl Send for Data {}
unsafe impl Sync for Data {}

impl Data {
    pub unsafe fn alloc(
        device: &Arc<CudaDevice>,
        data_type: DataType,
        len: usize,
    ) -> Result<Self, CudaError> {
        let num_bytes = len * data_type.size();
        let ptr = unsafe { driver::result::malloc_sync(num_bytes)? };
        Ok(Self {
            device: Arc::clone(device),
            ptr,
            typ: data_type,
        })
    }

    pub unsafe fn alloc_async(
        device: &Arc<CudaDevice>,
        stream: &CudaStream,
        data_type: DataType,
        len: usize,
    ) -> Result<Self, CudaError> {
        let num_bytes = len * data_type.size();
        let ptr = unsafe { driver::result::malloc_async(stream.raw(), num_bytes)? };
        Ok(Self {
            device: Arc::clone(device),
            ptr,
            typ: data_type,
        })
    }

    pub unsafe fn free_async(self, stream: &CudaStream) -> Result<(), CudaError> {
        unsafe {
            driver::result::free_async(self.ptr, stream.raw())?;
        }
        Ok(())
    }

    pub fn device_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    pub fn device_ptr_mut(&mut self) -> *mut u8 {
        self.ptr as *mut u8
    }

    pub fn data_type(&self) -> DataType {
        self.typ
    }
}

impl Drop for Data {
    fn drop(&mut self) {
        unsafe {
            driver::result::free_sync(self.ptr).ok();
        }
    }
}
