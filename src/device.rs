#[derive(Debug, Copy, Clone)]
pub enum Device {
    #[cfg(feature = "cuda")]
    Cuda(u32),
    #[cfg(feature = "cpu")]
    Cpu,
}
