#[derive(Debug, Clone)]
pub enum Device {
    #[cfg(feature = "cuda")]
    Cuda(u32),
}
