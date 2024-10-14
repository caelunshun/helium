use half::{bf16, f16};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DataType {
    F16,
    Bf16,
    F32,
}

impl DataType {
    pub fn size(self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 | DataType::Bf16 => 2,
        }
    }
}

pub trait DataTypeConversion: Copy + Sized {
    fn data_type() -> DataType;
    fn into_f32(self) -> f32;
    fn from_f32(x: f32) -> Self;
    fn into_data_vec(vec: Vec<Self>) -> DataVec;
}

impl DataTypeConversion for f32 {
    fn data_type() -> DataType {
        DataType::F32
    }

    fn into_f32(self) -> f32 {
        self
    }

    fn from_f32(x: f32) -> Self {
        x
    }

    fn into_data_vec(vec: Vec<Self>) -> DataVec {
        DataVec::F32(vec)
    }
}

impl DataTypeConversion for f16 {
    fn data_type() -> DataType {
        DataType::F16
    }

    fn into_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(x: f32) -> Self {
        Self::from_f32(x)
    }

    fn into_data_vec(vec: Vec<Self>) -> DataVec {
        DataVec::F16(vec)
    }
}

impl DataTypeConversion for bf16 {
    fn data_type() -> DataType {
        DataType::Bf16
    }

    fn into_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(x: f32) -> Self {
        Self::from_f32(x)
    }

    fn into_data_vec(vec: Vec<Self>) -> DataVec {
        DataVec::Bf16(vec)
    }
}

#[derive(Debug, Clone)]
pub enum DataVec {
    F32(Vec<f32>),
    Bf16(Vec<bf16>),
    F16(Vec<f16>),
}

impl DataVec {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            DataVec::F32(v) => bytemuck::cast_slice(v),
            DataVec::Bf16(v) => bytemuck::cast_slice(v),
            DataVec::F16(v) => bytemuck::cast_slice(v),
        }
    }
}
