#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    #[cfg(feature = "cuda")]
    Cuda(u32),
    #[cfg(feature = "cpu")]
    Cpu,
}
