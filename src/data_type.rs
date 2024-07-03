#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    Bf16,
    F16,
}

impl DataType {
    pub fn size(self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 | DataType::Bf16 => 2,
        }
    }
}
