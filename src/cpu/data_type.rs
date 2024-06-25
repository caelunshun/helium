use crate::{cpu::storage::PaddedChunk, data_type::DataType};
use bytemuck::Pod;
use half::{bf16, f16};
use std::simd::{prelude::SimdUint, u16x16};

/// Element type that can be stored in a `Storage`
/// and which can be read and written from memory
/// in CPU kernels.
pub trait CpuDataType: Copy + Pod {
    /// An array of `f32` whose length corresponds to the number
    /// of elements of `Self` that fit into a 64-byte chunk.
    type ChunkFloats: FloatArray;

    fn data_type() -> DataType;

    /// Decodes a chunk of `Self` into `f32` values
    /// for computation. Should round toward nearest even
    /// if `Self` has lower precision than `f32`.
    fn decode_chunk(chunk: &PaddedChunk) -> Self::ChunkFloats;

    /// Encodes a chunk of floats into the in-memory representation.
    fn encode_chunk(floats: Self::ChunkFloats) -> PaddedChunk;
}

pub trait FloatArray: Copy + Pod {
    fn as_slice(&self) -> &[f32];
}

impl<const N: usize> FloatArray for [f32; N] {
    fn as_slice(&self) -> &[f32] {
        self
    }
}

impl CpuDataType for f32 {
    type ChunkFloats = [f32; 16];

    fn data_type() -> DataType {
        DataType::F32
    }

    fn decode_chunk(chunk: &PaddedChunk) -> Self::ChunkFloats {
        bytemuck::cast(*chunk)
    }

    fn encode_chunk(floats: Self::ChunkFloats) -> PaddedChunk {
        bytemuck::cast(floats)
    }
}

impl CpuDataType for bf16 {
    type ChunkFloats = [f32; 32];

    fn data_type() -> DataType {
        DataType::Bf16
    }

    fn decode_chunk(chunk: &PaddedChunk) -> Self::ChunkFloats {
        decode_bf16(chunk)
    }

    fn encode_chunk(floats: Self::ChunkFloats) -> PaddedChunk {
        encode_bf16(floats)
    }
}

impl CpuDataType for f16 {
    type ChunkFloats = [f32; 32];

    fn data_type() -> DataType {
        DataType::F16
    }

    fn decode_chunk(chunk: &PaddedChunk) -> Self::ChunkFloats {
        decode_f16(chunk)
    }

    fn encode_chunk(floats: Self::ChunkFloats) -> PaddedChunk {
        encode_f16(floats)
    }
}

fn decode_bf16(chunk: &PaddedChunk) -> [f32; 32] {
    // `bf16 => f32` is achieved by filling in lower 16 bits with zeros.
    let bits: u16x16 = bytemuck::cast(*chunk);
    let bits = bits.cast::<u32>() << 16;
    bytemuck::cast(bits)
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bf16"
))]
fn encode_bf16(floats: [f32; 32]) -> PaddedChunk {
    use std::arch::x86_64::*;

    let bfloats = unsafe {
        let lower_floats = _mm512_loadu_ps(floats.as_ptr());
        let upper_floats = _mm512_loadu_ps(floats.as_ptr().add(16));
        _mm512_cvtne2ps_pbh(lower_floats, upper_floats)
    };
    bytemuck::cast(bfloats)
}

fn decode_f16(chunk: &PaddedChunk) -> [f32; 32] {
    use half::slice::HalfFloatSliceExt;
    let halfs: &[f16; 32] = bytemuck::cast_ref(chunk);
    let mut floats = [0.0; 32];
    halfs.convert_to_f32_slice(&mut floats);
    floats
}

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bf16"
)))]
fn encode_bf16(floats: [f32; 32]) -> PaddedChunk {
    use half::slice::HalfFloatSliceExt;
    let mut bfloats = [bf16::ZERO; 32];
    bfloats.convert_from_f32_slice(&floats);
    bytemuck::cast(bfloats)
}

fn encode_f16(floats: [f32; 32]) -> PaddedChunk {
    use half::slice::HalfFloatSliceExt;
    let mut halfs = [f16::ZERO; 32];
    halfs.convert_from_f32_slice(&floats);
    bytemuck::cast(halfs)
}
