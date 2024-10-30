use half::{bf16, f16};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DataType {
    F16,
    Bf16,
    F32,

    U32,

    Bool,
}

impl DataType {
    pub fn size_in_bits(self) -> usize {
        match self {
            DataType::F32 | DataType::U32 => 32,
            DataType::F16 | DataType::Bf16 => 16,
            DataType::Bool => 1,
        }
    }

    pub fn class(self) -> DataClass {
        match self {
            DataType::F32 | DataType::Bf16 | DataType::F16 => DataClass::Float,
            DataType::U32 => DataClass::Int,
            DataType::Bool => DataClass::Bool,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DataClass {
    Float,
    Int,
    Bool,
}

pub trait DataClassTrait {
    type HighP: Copy;

    fn data_class() -> DataClass;
}

#[derive(Copy, Clone, Debug)]
pub struct Float;
impl DataClassTrait for Float {
    type HighP = f64;

    fn data_class() -> DataClass {
        DataClass::Float
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Int;
impl DataClassTrait for Int {
    type HighP = i64;

    fn data_class() -> DataClass {
        DataClass::Int
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Bool;
impl DataClassTrait for Bool {
    type HighP = bool;

    fn data_class() -> DataClass {
        DataClass::Bool
    }
}

pub trait DataTypeConversion<C: DataClassTrait>: Copy + Sized {
    fn data_type() -> DataType;
    fn into_highp(self) -> C::HighP;
    fn from_highp(x: C::HighP) -> Self;
    fn into_data_vec(vec: Vec<Self>) -> DataVec;
}

impl DataTypeConversion<Float> for f32 {
    fn data_type() -> DataType {
        DataType::F32
    }

    fn into_highp(self) -> f64 {
        self.into()
    }

    fn from_highp(x: f64) -> Self {
        x as f32
    }

    fn into_data_vec(vec: Vec<Self>) -> DataVec {
        DataVec::F32(vec)
    }
}

impl DataTypeConversion<Float> for f16 {
    fn data_type() -> DataType {
        DataType::F16
    }

    fn into_highp(self) -> f64 {
        self.to_f64()
    }

    fn from_highp(x: f64) -> Self {
        Self::from_f64(x)
    }

    fn into_data_vec(vec: Vec<Self>) -> DataVec {
        DataVec::F16(vec)
    }
}

impl DataTypeConversion<Float> for bf16 {
    fn data_type() -> DataType {
        DataType::Bf16
    }

    fn into_highp(self) -> f64 {
        self.to_f64()
    }

    fn from_highp(x: f64) -> Self {
        Self::from_f64(x)
    }

    fn into_data_vec(vec: Vec<Self>) -> DataVec {
        DataVec::Bf16(vec)
    }
}

impl DataTypeConversion<Int> for u32 {
    fn data_type() -> DataType {
        DataType::U32
    }

    fn into_highp(self) -> i64 {
        self.into()
    }

    fn from_highp(x: i64) -> Self {
        x as u32
    }

    fn into_data_vec(vec: Vec<Self>) -> DataVec {
        DataVec::U32(vec)
    }
}

#[derive(Debug, Clone)]
pub enum DataVec {
    F32(Vec<f32>),
    Bf16(Vec<bf16>),
    F16(Vec<f16>),
    U32(Vec<u32>),
    /// Packed as bitset
    Bool(Vec<u32>),
}

impl DataVec {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            DataVec::F32(v) => bytemuck::cast_slice(v),
            DataVec::Bf16(v) => bytemuck::cast_slice(v),
            DataVec::F16(v) => bytemuck::cast_slice(v),
            DataVec::U32(v) => bytemuck::cast_slice(v),
            DataVec::Bool(v) => bytemuck::cast_slice(v),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            DataVec::F32(v) => v.len(),
            DataVec::Bf16(v) => v.len(),
            DataVec::F16(v) => v.len(),
            DataVec::U32(v) => v.len(),
            DataVec::Bool(v) => v.len() * u32::BITS as usize,
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            DataVec::F32(_) => DataType::F32,
            DataVec::Bf16(_) => DataType::Bf16,
            DataVec::F16(_) => DataType::F16,
            DataVec::U32(_) => DataType::U32,
            DataVec::Bool(_) => DataType::Bool,
        }
    }

    pub fn to_floats<T: DataTypeConversion<Float>>(&self) -> Vec<T> {
        match self {
            DataVec::F32(v) => v
                .iter()
                .copied()
                .map(f32::into_highp)
                .map(T::from_highp)
                .collect(),
            DataVec::F16(v) => v
                .iter()
                .copied()
                .map(f16::into_highp)
                .map(T::from_highp)
                .collect(),
            DataVec::Bf16(v) => v
                .iter()
                .copied()
                .map(bf16::into_highp)
                .map(T::from_highp)
                .collect(),
            _ => panic!(
                "called to_floats() on DataVec of non-float type {:?}",
                self.data_type()
            ),
        }
    }

    pub fn to_ints<T: DataTypeConversion<Int>>(&self) -> Vec<T> {
        match self {
            DataVec::U32(v) => v
                .iter()
                .copied()
                .map(u32::into_highp)
                .map(T::from_highp)
                .collect(),
            _ => panic!(
                "called to_ints() on DataVec of non-integer type {:?}",
                self.data_type()
            ),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum DataSlice<'a> {
    F32(&'a [f32]),
    Bf16(&'a [bf16]),
    F16(&'a [f16]),
    U32(&'a [u32]),
    /// Packed as bitset
    Bool(&'a [bool]),
}

impl<'a> DataSlice<'a> {
    pub fn as_bytes(&self) -> &'a [u8] {
        match self {
            DataSlice::F32(v) => bytemuck::cast_slice(v),
            DataSlice::Bf16(v) => bytemuck::cast_slice(v),
            DataSlice::F16(v) => bytemuck::cast_slice(v),
            DataSlice::U32(v) => bytemuck::cast_slice(v),
            DataSlice::Bool(v) => bytemuck::cast_slice(v),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            DataSlice::F32(v) => v.len(),
            DataSlice::Bf16(v) => v.len(),
            DataSlice::F16(v) => v.len(),
            DataSlice::U32(v) => v.len(),
            DataSlice::Bool(v) => v.len() * u32::BITS as usize,
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            DataSlice::F32(_) => DataType::F32,
            DataSlice::Bf16(_) => DataType::Bf16,
            DataSlice::F16(_) => DataType::F16,
            DataSlice::U32(_) => DataType::U32,
            DataSlice::Bool(_) => DataType::Bool,
        }
    }
}

impl<'a> From<&'a [f32]> for DataSlice<'a> {
    fn from(value: &'a [f32]) -> Self {
        Self::F32(value)
    }
}

impl<'a> From<&'a [bf16]> for DataSlice<'a> {
    fn from(value: &'a [bf16]) -> Self {
        Self::Bf16(value)
    }
}

impl<'a> From<&'a [f16]> for DataSlice<'a> {
    fn from(value: &'a [f16]) -> Self {
        Self::F16(value)
    }
}

impl<'a> From<&'a [u32]> for DataSlice<'a> {
    fn from(value: &'a [u32]) -> Self {
        Self::U32(value)
    }
}

pub trait AsDataSlice {
    fn as_data_slice(&self) -> DataSlice;
}

impl AsDataSlice for &'_ [f32] {
    fn as_data_slice(&self) -> DataSlice {
        DataSlice::F32(self)
    }
}

impl AsDataSlice for &'_ [f16] {
    fn as_data_slice(&self) -> DataSlice {
        DataSlice::F16(self)
    }
}

impl AsDataSlice for &'_ [bf16] {
    fn as_data_slice(&self) -> DataSlice {
        DataSlice::Bf16(self)
    }
}

impl AsDataSlice for &'_ [u32] {
    fn as_data_slice(&self) -> DataSlice {
        DataSlice::U32(self)
    }
}

impl AsDataSlice for Vec<f32> {
    fn as_data_slice(&self) -> DataSlice {
        DataSlice::F32(self.as_slice())
    }
}

impl AsDataSlice for Vec<bf16> {
    fn as_data_slice(&self) -> DataSlice {
        DataSlice::Bf16(self.as_slice())
    }
}

impl AsDataSlice for Vec<f16> {
    fn as_data_slice(&self) -> DataSlice {
        DataSlice::F16(self.as_slice())
    }
}

impl AsDataSlice for Vec<u32> {
    fn as_data_slice(&self) -> DataSlice {
        DataSlice::U32(self.as_slice())
    }
}
