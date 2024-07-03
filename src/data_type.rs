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
