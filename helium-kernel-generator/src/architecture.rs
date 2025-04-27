/// SM architecture version.
///
/// "a" suffixes mean that kernels generated for that architecture
/// are not forward compatible with later architectures.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Architecture {
    /// Ampere data center (A100). NVIDIA doesn't call this "SM80a,"
    /// but it has 192KB shared memory per SMEM which makes
    /// kernels generated for it architecture-specific.
    Sm80a,
    /// Baseline architecture (corresponds to consumer RTX 30xx series,
    /// and A40/A6000 etc. enterprise GPUs).
    Sm86,
    /// Ada Lovelace (RTX 40xx, L40S, RTX 6000 Ada, etc.)
    Sm89,
    /// Hopper (H100/H200)
    Sm90a,
    /// Blackwell data center (B100/B200)
    Sm100a,
    /// Blackwell consumer (RTX 50xx, RTX PRO Blackwell)
    Sm120a,
}

impl Architecture {
    /// Maximum size of shared memory that can be used by a kernel
    /// (requires dynamic shared memory).
    pub(crate) fn max_smem_size(&self) -> u32 {
        match self {
            Architecture::Sm86 | Architecture::Sm89 | Architecture::Sm120a => 99 * 1024,
            Architecture::Sm80a => 163 * 1024,
            Architecture::Sm90a | Architecture::Sm100a => 227 * 1024,
        }
    }

    pub(crate) fn nvrtc_flag(&self) -> String {
        let name = match self {
            Architecture::Sm80a => "sm_80",
            Architecture::Sm86 => "sm_80", // intentional; sm_86 isn't understood by nvcc
            Architecture::Sm89 => "sm_89",
            Architecture::Sm90a => "sm_90a",
            Architecture::Sm100a => "sm_100a",
            Architecture::Sm120a => "sm_120a",
        };
        format!("-arch={name}")
    }
}
