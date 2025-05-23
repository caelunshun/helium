use crate::cute::{Layout, Mode};
use std::fmt::{Display, Formatter};

#[derive(Debug, PartialEq, Eq)]
pub struct CopyPattern {
    pub thread_layout: Layout,
    pub value_layout: Layout,
    pub vectorization_type: CopyVectorizationType,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CopyVectorizationType {
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Uint128,
}

impl CopyVectorizationType {
    pub fn from_byte_size(byte_size: u32) -> Option<Self> {
        match byte_size {
            1 => Some(Self::Uint8),
            2 => Some(Self::Uint16),
            4 => Some(Self::Uint32),
            8 => Some(Self::Uint64),
            16 => Some(Self::Uint128),
            _ => None,
        }
    }

    pub fn byte_size(self) -> u32 {
        match self {
            CopyVectorizationType::Uint8 => 1,
            CopyVectorizationType::Uint16 => 2,
            CopyVectorizationType::Uint32 => 4,
            CopyVectorizationType::Uint64 => 8,
            CopyVectorizationType::Uint128 => 16,
        }
    }

    pub fn with_alignment_restriction(self, align: u32) -> Self {
        match self {
            CopyVectorizationType::Uint128 if align < 16 => {
                CopyVectorizationType::Uint64.with_alignment_restriction(align)
            }
            CopyVectorizationType::Uint64 if align < 8 => {
                CopyVectorizationType::Uint32.with_alignment_restriction(align)
            }
            CopyVectorizationType::Uint32 if align < 4 => {
                CopyVectorizationType::Uint16.with_alignment_restriction(align)
            }
            CopyVectorizationType::Uint16 if align < 2 => CopyVectorizationType::Uint8,
            _ => self,
        }
    }
}

/// Returns the greatest power of 2 that evenly divides `x`.
fn alignment_of(x: u32) -> u32 {
    2u32.pow(x.trailing_zeros())
}

impl Display for CopyVectorizationType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CopyVectorizationType::Uint8 => write!(f, "uint8_t"),
            CopyVectorizationType::Uint16 => write!(f, "uint16_t"),
            CopyVectorizationType::Uint32 => write!(f, "uint32_t"),
            CopyVectorizationType::Uint64 => write!(f, "uint64_t"),
            CopyVectorizationType::Uint128 => write!(f, "uint128_t"),
        }
    }
}

/// Finds a copy pattern for a tile in global memory
/// that aims to maximize load/store coalescing.
pub fn find_gmem_copy_pattern(
    gmem_layout: &Layout,
    tile_layout: &Layout,
    num_threads: u32,
    bytes_per_element: u32,
) -> CopyPattern {
    let gmem_modes: Vec<_> = gmem_layout.nested().collect();
    let tile_modes: Vec<_> = tile_layout.nested().collect();

    assert_eq!(
        gmem_modes.len(),
        tile_modes.len(),
        "number of outermost dimensions in gmem and in tile must match"
    );
    assert!(num_threads >= 32, "at least one warp required");
    assert!(
        num_threads.is_power_of_two(),
        "thread count participating in copy must be a power of two"
    );

    assert!(
        tile_modes.iter().all(|mode| mode.size().is_power_of_two()),
        "tile dimensions must be powers of two (for now)"
    );

    assert!(bytes_per_element.is_power_of_two());

    // Strategy - match the ordering of strides
    // in the gmem layout to the ordering of strides
    // in the thread layout to maximize locality of data
    // within warps, thereby maximizing coalescing.
    // For nested layout modes, we currently use
    // the minimum stride of all the modes as a heuristic.
    let gmem_strides: Vec<_> = gmem_modes
        .iter()
        .map(|mode| {
            mode.flatten()
                .into_iter()
                .map(|mode| mode.stride)
                .min()
                .unwrap()
        })
        .collect();
    let mut order = (0..gmem_modes.len()).collect::<Vec<_>>();
    order.sort_by_key(|i| gmem_strides[*i]);

    let mut thread_layout = vec![Mode::default(); gmem_modes.len()];
    let mut value_layout = vec![Mode::default(); gmem_modes.len()];

    let mut threads_stride = 1;
    let mut values_stride = 1;
    for (i, mode_index) in order.iter().copied().enumerate() {
        thread_layout[mode_index].stride = threads_stride;
        // For the major dimension, we want to satisfy the following criteria:
        // 1. Each thread should load at least 4 bytes if possible, so that we
        // can vectorize to at least uint32_t - otherwise, we would be using
        // an inefficient narrow load instruction (hardware uses 128-byte coalescing, 4x32 = 128)
        // 2. Each thread should load no more than 16 bytes along that dimension,
        // or else the compiler will generate strided instructions that will kill performance.
        // For the remaining dimensions, the thread/value count typically does not affect performance
        // except in extreme cases where e.g. the second-major dimension is very small.
        let threads_size = match i {
            // major dimension
            0 => {
                let major_bytes = tile_modes[mode_index].size() * bytes_per_element;
                let mut threads = (major_bytes / 16).min(num_threads); // 16 == max vectorization size, sizeof(uint128_t)
                // edge case: if there would be too many remaining threads
                // to fill the rest of the dimensions, then we have to increase
                // the count (meaning worse vectorization)
                let remaining_dimensions = order[1..]
                    .iter()
                    .map(|&i| tile_modes[i].size())
                    .product::<u32>();
                if num_threads / threads > remaining_dimensions {
                    threads *= (num_threads / threads) / remaining_dimensions;
                }
                threads
            }
            // remaining dimensions
            _ => (num_threads / threads_stride).max(1),
        };
        thread_layout[mode_index].size = threads_size;
        threads_stride *= threads_size;

        value_layout[mode_index].stride = values_stride;
        let values_size = tile_modes[mode_index].size() / threads_size;
        value_layout[mode_index].size = values_size;
        values_stride *= values_size;
    }

    // Determine vectorization type
    let major_value_size = value_layout[order[0]].size;
    let bytes_per_thread = major_value_size * bytes_per_element;
    let vectorization_type =
        CopyVectorizationType::from_byte_size(bytes_per_thread.min(16)).unwrap();

    // If global memory or shared memory cannot be proven to be aligned to
    // the vectorization type, then we need to use a narrower
    // vectorization. The proven alignment equals the minimum
    // of the alignments of the strides of the modes except the major mode.
    let mut alignment = u32::MAX;
    for (i, mode) in gmem_modes.iter().enumerate() {
        if i != order[0] {
            for nested_mode in mode.flatten() {
                alignment = alignment.min(alignment_of(nested_mode.stride * bytes_per_element));
            }
        }
    }
    for (i, mode) in tile_modes.iter().enumerate() {
        if i != order[0] {
            for nested_mode in mode.flatten() {
                alignment = alignment.min(alignment_of(nested_mode.stride * bytes_per_element));
            }
        }
    }
    let vectorization_type = vectorization_type.with_alignment_restriction(alignment);

    // If achieved vectorization width is less than the value size along the major axis,
    // then this would lead to strided instructions, killing performance.
    // In that case, we try to redistribute the values according to the vectorization
    // type.
    if vectorization_type.byte_size() < value_layout[order[0]].size * bytes_per_element {
        let fewer_values_per_thread_factor =
            value_layout[order[0]].size * bytes_per_element / vectorization_type.byte_size();
        value_layout[order[0]].size /= fewer_values_per_thread_factor;
        thread_layout[order[0]].size *= fewer_values_per_thread_factor;

        // there are now fewer threads along the remaining axes, so we have to divide
        // them out while increasing the value sizes along the remaining axes.
        let mut extra_threads = fewer_values_per_thread_factor;
        for &i in &order[1..] {
            if extra_threads == 1 {
                break;
            }

            let divisor = gcd::binary_u32(extra_threads, thread_layout[i].size);
            thread_layout[i].size /= divisor;
            value_layout[i].size *= divisor;
            extra_threads /= divisor;
        }

        // update strides
        let mut threads_stride = 1;
        let mut values_stride = 1;
        for &i in &order {
            thread_layout[i].stride = threads_stride;
            value_layout[i].stride = values_stride;
            threads_stride *= thread_layout[i].size;
            values_stride *= value_layout[i].size;
        }
    }

    for &i in &order[1..] {
        value_layout[i].size = 1;
    }

    CopyPattern {
        thread_layout: Layout::MultiMode(
            thread_layout.into_iter().map(Layout::SingleMode).collect(),
        )
        .normalized(),
        value_layout: Layout::MultiMode(value_layout.into_iter().map(Layout::SingleMode).collect())
            .normalized(),
        vectorization_type,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_calculation() {
        assert_eq!(alignment_of(256), 256);
        assert_eq!(alignment_of(257), 1);
        assert_eq!(alignment_of(258), 2);
        assert_eq!(alignment_of(384), 128);
    }

    #[test]
    fn col_major_copy_pattern() {
        assert_eq!(
            find_gmem_copy_pattern(
                &Layout::new_column_major(&[256, 256]),
                &Layout::new_column_major(&[128, 64]),
                256,
                4
            ),
            CopyPattern {
                thread_layout: Layout::new_column_major(&[32, 8]),
                value_layout: Layout::new_column_major(&[4, 1]),
                vectorization_type: CopyVectorizationType::Uint128,
            }
        );
    }

    #[test]
    fn row_major_copy_pattern() {
        assert_eq!(
            find_gmem_copy_pattern(
                &Layout::new_row_major(&[256, 256]),
                &Layout::new_row_major(&[128, 64]),
                256,
                4
            ),
            CopyPattern {
                thread_layout: Layout::new_row_major(&[16, 16]),
                value_layout: Layout::new_row_major(&[1, 4]),
                vectorization_type: CopyVectorizationType::Uint128,
            }
        );
    }

    #[test]
    fn narrow_dimension_copy_pattern() {
        assert_eq!(
            find_gmem_copy_pattern(
                &Layout::new_column_major(&[256, 256]),
                &Layout::new_column_major(&[32, 64]),
                256,
                2
            ),
            CopyPattern {
                thread_layout: Layout::new_column_major(&[4, 64]),
                value_layout: Layout::new_column_major(&[8, 1]),
                vectorization_type: CopyVectorizationType::Uint128,
            }
        );
    }

    #[test]
    fn misalignment_inhibits_copy_vectorization() {
        assert_eq!(
            find_gmem_copy_pattern(
                &Layout::new_column_major(&[257, 256]),
                &Layout::new_column_major(&[32, 64]),
                256,
                2
            ),
            CopyPattern {
                thread_layout: Layout::new_column_major(&[32, 8]),
                value_layout: Layout::new_column_major(&[1, 1]),
                vectorization_type: CopyVectorizationType::Uint16,
            }
        );
    }

    #[test]
    fn very_narrow_dimension_copy_pattern() {
        assert_eq!(
            find_gmem_copy_pattern(
                &Layout::new_column_major(&[256, 256]),
                &Layout::new_column_major(&[32, 32]),
                256,
                2
            ),
            CopyPattern {
                thread_layout: Layout::new_column_major(&[8, 32]),
                value_layout: Layout::new_column_major(&[4, 1]),
                vectorization_type: CopyVectorizationType::Uint64,
            }
        );
    }

    #[test]
    fn transposed_smem_layout_inhibits_vectorization() {
        assert_eq!(
            find_gmem_copy_pattern(
                &Layout::new_column_major(&[256, 256]),
                &Layout::new_row_major(&[128, 64]),
                256,
                4
            ),
            CopyPattern {
                thread_layout: Layout::new_column_major(&[128, 2]),
                value_layout: Layout::new_column_major(&[1, 1]),
                vectorization_type: CopyVectorizationType::Uint32,
            }
        );
    }

    #[test]
    fn smem_layout_affects_vectorization() {
        assert_eq!(
            find_gmem_copy_pattern(
                &Layout::new_row_major(&[256, 256]),
                &Layout::new_column_major(&[128, 32]),
                256,
                2
            ),
            CopyPattern {
                thread_layout: Layout::new_row_major(&[8, 32]),
                value_layout: Layout::new_row_major(&[1, 1]),
                vectorization_type: CopyVectorizationType::Uint16,
            }
        );
    }
}
