use crate::{
    cuda::{context::CudaStream, error::CudaError},
    data_type::{DataType, DataTypeConversion},
    shape::Shape,
};
use cudarc::{
    driver,
    driver::{sys::CUdeviceptr, CudaDevice},
};
use half::{bf16, f16};
use std::{mem, ptr, sync::Arc};

/// Raw CUDA tensor.
#[derive(Clone)]
pub struct RawTensor {
    shape: Shape,
    data: Arc<Data>,
}

impl RawTensor {
    pub fn new(data: Data, shape: Shape) -> Self {
        Self {
            shape,
            data: Arc::new(data),
        }
    }

    pub fn reshaped(&self, new_shape: Shape) -> Self {
        assert_eq!(new_shape.num_elements(), self.num_elements());
        Self {
            data: Arc::clone(&self.data),
            shape: new_shape,
        }
    }

    pub fn fill(&mut self, val: f32, stream: &CudaStream) -> Result<(), CudaError> {
        match self.data_type() {
            DataType::F32 => unsafe {
                driver::sys::lib().cuMemsetD32Async(
                    self.data.ptr,
                    val.to_bits(),
                    self.num_elements(),
                    stream.raw(),
                );
            },
            DataType::Bf16 => unsafe {
                driver::sys::lib().cuMemsetD16Async(
                    self.data.ptr,
                    bf16::from_f32(val).to_bits(),
                    self.num_elements(),
                    stream.raw(),
                );
            },
            DataType::F16 => unsafe {
                driver::sys::lib().cuMemsetD16Async(
                    self.data.ptr,
                    f16::from_f32(val).to_bits(),
                    self.num_elements(),
                    stream.raw(),
                );
            },
        }
        Ok(())
    }

    pub fn dimension(&self) -> u32 {
        self.shape.num_dims() as u32
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dim_at(&self, x: i32) -> usize {
        self.shape.dim_at(x)
    }

    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    pub fn data(&self) -> &Data {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut Data {
        Arc::get_mut(&mut self.data).unwrap()
    }

    pub fn data_type(&self) -> DataType {
        self.data.data_type()
    }

    pub fn into_data(self) -> Data {
        Arc::into_inner(self.data).unwrap()
    }

    pub fn from_slice_async<T: DataTypeConversion>(
        slice: &[T],
        data_type: DataType,
        shape: Shape,
        stream: &CudaStream,
        device: &Arc<CudaDevice>,
    ) -> Result<Self, CudaError> {
        let len = shape.num_elements();
        let data = unsafe { Data::alloc_async(device, stream, data_type, len)? };
        match data_type {
            DataType::F16 => {
                let vec: Vec<f16> = slice
                    .iter()
                    .copied()
                    .map(|x| f16::from_f32(x.into_f32()))
                    .collect();
                unsafe {
                    driver::result::memcpy_htod_async(data.ptr, &vec, stream.raw())?;
                }
            }
            DataType::Bf16 => {
                let vec: Vec<bf16> = slice
                    .iter()
                    .copied()
                    .map(|x| bf16::from_f32(x.into_f32()))
                    .collect();
                unsafe {
                    driver::result::memcpy_htod_async(data.ptr, &vec, stream.raw())?;
                }
            }
            DataType::F32 => {
                let vec: Vec<f32> = slice.iter().copied().map(T::into_f32).collect();
                unsafe {
                    driver::result::memcpy_htod_async(data.ptr, &vec, stream.raw())?;
                }
            }
        };

        Ok(Self::new(data, shape))
    }

    pub fn to_vec_sync<T: DataTypeConversion>(&self) -> Result<Vec<T>, CudaError> {
        match self.data_type() {
            DataType::F16 => {
                let mut buf = vec![f16::ZERO; self.num_elements()];
                unsafe {
                    driver::result::memcpy_dtoh_sync(&mut buf, self.data.ptr)?;
                }
                Ok(buf.into_iter().map(|x| T::from_f32(x.into_f32())).collect())
            }
            DataType::Bf16 => {
                let mut buf = vec![bf16::ZERO; self.num_elements()];
                unsafe {
                    driver::result::memcpy_dtoh_sync(&mut buf, self.data.ptr)?;
                }
                Ok(buf.into_iter().map(|x| T::from_f32(x.into_f32())).collect())
            }
            DataType::F32 => {
                let mut buf = vec![0.0f32; self.num_elements()];
                unsafe {
                    driver::result::memcpy_dtoh_sync(&mut buf, self.data.ptr)?;
                }
                Ok(buf.into_iter().map(T::from_f32).collect())
            }
        }
    }
}

pub struct Data {
    ptr: CUdeviceptr,
    typ: DataType,
}

unsafe impl Send for Data {}
unsafe impl Sync for Data {}

impl Data {
    pub unsafe fn alloc_async(
        _device: &CudaDevice,
        stream: &CudaStream,
        data_type: DataType,
        len: usize,
    ) -> Result<Self, CudaError> {
        let num_bytes = len * data_type.size();
        let ptr = unsafe { driver::result::malloc_async(stream.raw(), num_bytes)? };
        Ok(Self {
            ptr,
            typ: data_type,
        })
    }

    pub unsafe fn free_async(self, stream: &CudaStream) -> Result<(), CudaError> {
        unsafe {
            driver::result::free_async(self.ptr, stream.raw())?;
        }
        mem::forget(self);
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
            driver::result::free_async(self.ptr, ptr::null_mut()).ok();
        }
    }
}
