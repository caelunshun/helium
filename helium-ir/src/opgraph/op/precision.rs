use crate::data_type::DataType;

/// Specifies the minimum precision of an operation that involves
/// dot product computation (i.e. matrix multiplication and convolution).
///
/// Note that depending on hardware features, the actual precision
/// used may be greater than the precision specified. However,
/// we guarantee it will never be less. Regardless of precision,
/// there is no guarantee that floating point arithmetic will be deterministic,
/// even when running on the same hardware.
///
/// There are two components to precision: the precision used for multiplications
/// and the precision used for accumulations (additions). Often,
/// using a lower precision for multiplication than accumulation leads
/// to a high performance improvement without much accuracy loss.
/// Lower precision accumulations tend to be riskier.
///
/// Precision has a strong impact on performance. Generally, on supported
/// hardware, using half the number of bits per element leads to twice the throughput.
/// The documentation comments for each precision setting provide
/// relative throughput values for NVIDIA GPUs; these values were
/// derived empirically with equivalent results on sm_80, sm_89, sm_90, sm_100, and sm_120
/// except where noted.
///
/// On CPUs, precision usually has no effect as most CPUs support no lower
/// than 32-bit FMA. The exception is CPUs that have AVX512-FP16 or possibly AVX512-BF16.
///
/// Note that the precision used for a matmul is independent of the input types
/// to the `Matmul` operator (or equivalent for convolution). Backends will automatically
/// insert conversions to the multiplicand data type when generating their kernels.
///
/// The output type of `Matmul` (resp. convolution) operators in the `OpGraph` is the accumulation
/// type. The high-level `Tensor` API has a different behavior with data types; see its documentation
/// for details.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Precision {
    /// Full 32-bit multiplication and accumulation.
    /// This will generally disable the use of dedicated
    /// matrix multiplication hardware (e.g. tensor cores), resulting in much
    /// slower throughput than all other settings. Not recommended
    /// for most deep learning tasks.
    MulF32AccumF32,
    /// Multiplication using 19-bit "tensorfloat32" (tf32),
    /// which combines the precision of `f16` with the dynamic
    /// range of `bf16`. Accumulation in `f32`.
    ///
    /// Relative throughput on NVIDIA: 1x
    MulTf32AccumF32,
    /// Multiplication using `bf16`. Accumulation in `f32`.
    ///
    /// Relative throughput on NVIDIA: 2x
    MulBf16AccumF32,
    /// Multiplication using `f16`. Accumulation in `f32`.
    ///
    /// Relative throughput on NVIDIA: 2x
    MulF16AccumF32,
    /// Multiplication using `f16`. Accumulation in `f16`.
    ///
    /// Relative throughput on NVIDIA: 4x
    MulF16AccumF16,
    /// Multiplication using 8-bit floating point values.
    /// Accumulation in `f32`.
    ///
    /// Relative throughput on NVIDIA: 4x (requires sm_89 or later)
    MulF8AccumF32 {
        /// Type of 8-bit precision to use for the A matrix.
        mode_a: F8Mode,
        /// Type of 8-bit precision to use for the B matrix.
        mode_b: F8Mode,
    },
    /// Multiplication using 8-bit floating point values.
    /// Accumulation in `f16`.
    ///
    /// Relative throughput on NVIDIA: 8x (requires sm_89 or later)
    MulF8AccumF16 {
        /// Type of 8-bit precision to use for the A matrix.
        mode_a: F8Mode,
        /// Type of 8-bit precision to use for the B matrix.
        mode_b: F8Mode,
    },
}

impl Precision {
    /// Gets the type used for the accumulation operation,
    /// which determines the output type of `Matmul` and `Conv`
    /// operators.
    pub fn accumulator_type(&self) -> DataType {
        match self {
            Precision::MulF32AccumF32
            | Precision::MulTf32AccumF32
            | Precision::MulBf16AccumF32
            | Precision::MulF16AccumF32
            | Precision::MulF8AccumF32 { .. } => DataType::F32,
            Precision::MulF16AccumF16 | Precision::MulF8AccumF16 { .. } => DataType::F16,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum F8Mode {
    /// 4-bit exponent, 3-bit mantissa.
    /// (Prefers precision rather than dynamic range.)
    E4M3,
    /// 5-bit exponent, 2-bit mantissa.
    /// (Prefers dynamic range rather than precision.)
    E5M2,
}
