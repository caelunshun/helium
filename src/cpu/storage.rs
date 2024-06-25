use crate::{cpu::CpuDataType, data_type::DataType};
use bytemuck::{Pod, Zeroable};
use half::{bf16, f16};

/// Raw tensor storage.
///
/// This container provides a few guarantees:
/// 1. The buffer is aligned to 64 bytes and has a size
///    padded to be a multiple of 64 bytes. This alignment corresponds
///    to the maximum alignment required by any current SIMD instruction set
///    (i.e. AVX-512) and ensures we get optimal performance during iteration.
/// 2. All values are initialized, including the "padding" values
///    at the end of the buffer.
#[derive(Debug)]
pub struct Storage {
    buffer: Vec<PaddedChunk>,
    /// Actual length of the buffer, i.e. the number
    /// of elements excluding padding elements.
    len: usize,
    data_type: DataType,
}

impl Storage {
    /// Creates a storage for `len` elements
    /// of type `T`. Values are initialized to zero.
    pub fn new<T: CpuDataType>(len: usize) -> Self {
        let min_num_bytes = len.checked_mul(size_of::<T>()).unwrap();
        let num_bytes = (min_num_bytes + 63) / 64 * 64;
        let num_chunks = num_bytes / 64;

        Self {
            buffer: vec![PaddedChunk::zeroed(); num_chunks],
            len,
            data_type: T::data_type(),
        }
    }

    /// Returns the data type of the storage.
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// # Panics
    /// Panics if the data type is not f32.
    pub fn as_f32(&self) -> &[f32] {
        assert_eq!(self.data_type, DataType::F32);
        bytemuck::cast_slice(&self.buffer)
    }

    /// # Panics
    /// Panics if the data type is not bf16.
    pub fn as_bf16(&self) -> &[bf16] {
        assert_eq!(self.data_type, DataType::Bf16);
        bytemuck::cast_slice(&self.buffer)
    }

    /// # Panics
    /// Panics if the data type is not f16.
    pub fn as_f16(&self) -> &[f16] {
        assert_eq!(self.data_type, DataType::F16);
        bytemuck::cast_slice(&self.buffer)
    }

    pub fn raw(&self) -> &[PaddedChunk] {
        &self.buffer
    }
}

#[derive(Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C, align(64))]
pub struct PaddedChunk(pub [u8; 64]);
