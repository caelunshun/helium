use helium_ir::shape::Shape;
use slotmap::Key;
use std::fmt::{Debug, Display, Formatter, Write};

/// Rust runtime representation mirroring `cute::Layout`
/// from CUTLASS.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Layout {
    SingleMode { size: u32, stride: u32 },
    MultiMode(Vec<Layout>),
}

impl Layout {
    /// Row-major layout from a tensor shape, densely packed,
    /// as standard in `helium`.
    pub fn from_tensor_shape(shape: &Shape) -> Self {
        let mut strides = Vec::new();
        let mut stride = 1;
        for dim in shape.dims().iter().copied().rev() {
            strides.push(stride);
            stride *= dim;
        }
        strides.reverse();

        Self::MultiMode(
            shape
                .dims()
                .iter()
                .copied()
                .zip(strides)
                .map(|(size, stride)| Layout::SingleMode {
                    size: size.try_into().unwrap(),
                    stride: stride.try_into().unwrap(),
                })
                .collect(),
        )
    }

    pub fn from_sizes_and_strides(sizes_strides: impl IntoIterator<Item = (u32, u32)>) -> Self {
        Self::MultiMode(
            sizes_strides
                .into_iter()
                .map(|(size, stride)| Layout::SingleMode { size, stride })
                .collect(),
        )
        .normalized()
    }

    pub fn new_column_major(sizes: &[u32]) -> Self {
        let mut stride = 1;
        Layout::MultiMode(
            sizes
                .iter()
                .copied()
                .map(|size| {
                    let mode = Layout::SingleMode { stride, size };
                    stride *= size;
                    mode
                })
                .collect(),
        )
        .normalized()
    }

    /// Removes unnecessary levels of nesting.
    #[must_use]
    pub fn normalized(self) -> Self {
        match self {
            Self::MultiMode(mut v) if v.len() == 1 => v.remove(0).normalized(),
            Self::MultiMode(v) => Self::MultiMode(v.into_iter().map(Self::normalized).collect()),
            Self::SingleMode { .. } => self,
        }
    }

    /// Functional composition of `self` with `other`, such
    /// that `ther` is applied, then `self`. i.e., `self(other(x))`
    pub fn compose(&self, other: &Self) -> Self {
        let a = self.flatten();
        let b = other.flatten();

        // https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md
        // "Computing Composition"

        let mut output_modes = Vec::new();

        for &(b_size, b_stride) in &b {
            let mut c = a.clone();
            // "remove" first b_stride elements in `c` by dividing them out
            let mut d = b_stride;
            for (c_size, c_stride) in &mut c {
                if d <= 1 {
                    break;
                }

                let divisor = d.min(*c_size);
                assert_eq!(*c_size % divisor, 0, "stride divisibility condition failed");
                *c_size /= divisor;
                *c_stride *= divisor;
                d /= divisor;
            }
            assert_eq!(d, 1, "stride divisibility condition failed");

            // "keep" first b_size elements in `c` by modding them out
            let s = b_size;
            let q = c.iter().copied().map(|(size, _)| size).product::<u32>();
            assert_eq!(q % s, 0, "size divisibility condition failed");
            let mut d = q / s;
            for (c_size, c_stride) in c.iter_mut().rev() {
                if d <= 1 {
                    break;
                }

                let divisor = d.min(*c_size);
                assert_eq!(*c_size % divisor, 0, "size divisibility condition failed");
                *c_size /= divisor;
                d /= divisor;
            }
            assert_eq!(d, 1, "size divisibility condition failed");

            output_modes.push(
                Layout::MultiMode(
                    c.into_iter()
                        .map(|(size, stride)| Layout::SingleMode { size, stride })
                        .collect(),
                )
                .coalesce(),
            );
        }

        Layout::MultiMode(output_modes).normalized()
    }

    /// Returns the C++ type of the corresponding `cute::Layout`
    /// template instance.
    pub fn cute_type(&self) -> String {
        fn build_shape(s: &mut String, layout: &Layout) {
            match layout {
                Layout::SingleMode { size, .. } => {
                    write!(s, "Int<{size}>").unwrap();
                }
                Layout::MultiMode(v) => {
                    write!(s, "Shape<").unwrap();
                    for (i, mode) in v.iter().enumerate() {
                        build_shape(s, mode);
                        if i != v.len() - 1 {
                            write!(s, ", ").unwrap();
                        }
                    }
                    write!(s, ">").unwrap();
                }
            }
        }

        fn build_stride(s: &mut String, layout: &Layout) {
            match layout {
                Layout::SingleMode { stride, .. } => {
                    write!(s, "Int<{stride}>").unwrap();
                }
                Layout::MultiMode(v) => {
                    write!(s, "Stride<").unwrap();
                    for (i, mode) in v.iter().enumerate() {
                        build_stride(s, mode);
                        if i != v.len() - 1 {
                            write!(s, ", ").unwrap();
                        }
                    }
                    write!(s, ">").unwrap();
                }
            }
        }

        let mut s = String::new();
        write!(s, "Layout<").unwrap();
        build_shape(&mut s, self);
        write!(s, ", ").unwrap();
        build_stride(&mut s, self);
        write!(s, ">").unwrap();
        s
    }

    fn flatten(&self) -> Vec<(u32, u32)> {
        fn flatten_into(l: &Layout, v: &mut Vec<(u32, u32)>) {
            match l {
                Layout::SingleMode { size, stride } => v.push((*size, *stride)),
                Layout::MultiMode(ls) => {
                    for l in ls {
                        flatten_into(l, v)
                    }
                }
            }
        }
        let mut v = Vec::new();
        flatten_into(self, &mut v);
        v
    }

    fn coalesce(&self) -> Self {
        let mut flattened = self.flatten();
        while flattened.len() > 1 {
            let mut did_coalesce = false;

            for i in 0..flattened.len() - 1 {
                let (size_a, stride_a) = flattened[i];
                let (size_b, stride_b) = flattened[i + 1];

                if size_a == 1 {
                    // ignore modes with size 1
                    flattened.remove(i);
                } else if size_b == 1 {
                    // ignore modes with size 1
                    flattened.remove(i + 1);
                } else if stride_b == stride_a * size_a {
                    flattened.remove(i);
                    flattened[i] = (size_a * size_b, stride_a);
                } else {
                    continue;
                }

                did_coalesce = true;
                break;
            }

            if !did_coalesce {
                break;
            }
        }
        Self::MultiMode(
            flattened
                .into_iter()
                .map(|(size, stride)| Self::SingleMode { size, stride })
                .collect(),
        )
        .normalized()
    }

    pub fn size(&self) -> u32 {
        match self {
            Layout::SingleMode { size, .. } => *size,
            Layout::MultiMode(v) => v.iter().map(Layout::size).product(),
        }
    }
}

impl Debug for Layout {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.cute_type())
    }
}

impl Display for Layout {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.cute_type())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn column_major_to_cute_type() {
        let layout = Layout::new_column_major(&[8, 4]);
        assert_eq!(
            layout.cute_type(),
            "Layout<Shape<Int<8>, Int<4>>, Stride<Int<1>, Int<8>>>"
        );
    }

    #[test]
    fn tensor_shape_to_cute_type() {
        let layout = Layout::from_tensor_shape(&Shape::new([256, 8]));
        assert_eq!(
            layout.cute_type(),
            "Layout<Shape<Int<256>, Int<8>>, Stride<Int<8>, Int<1>>>"
        );
    }

    #[test]
    fn nested_layout_to_cute_type() {
        let layout = Layout::MultiMode(vec![
            Layout::MultiMode(vec![
                Layout::SingleMode { size: 4, stride: 1 },
                Layout::SingleMode { size: 8, stride: 4 },
            ]),
            Layout::SingleMode {
                size: 16,
                stride: 32,
            },
        ]);
        assert_eq!(
            layout.cute_type(),
            "Layout<Shape<Shape<Int<4>, Int<8>>, Int<16>>, Stride<Stride<Int<1>, Int<4>>, Int<32>>>"
        );
    }

    #[rstest]
    #[case(
        Layout::from_sizes_and_strides([(4, 2)]),
        Layout::from_sizes_and_strides([(2, 2)]),
        Layout::from_sizes_and_strides([(2, 4)]))]
    #[case(
        Layout::from_sizes_and_strides([(6, 8), (2, 2)]),
        Layout::from_sizes_and_strides([(4, 3), (3, 1)]),
        Layout::MultiMode(vec![Layout::from_sizes_and_strides([(2, 24), (2, 2)]), Layout::from_sizes_and_strides([(3, 8)])]
        ))]
    #[case(
        Layout::from_sizes_and_strides([(10, 16), (2, 4)]),
        Layout::from_sizes_and_strides([(5, 1), (4, 5)]),
        Layout::MultiMode(vec![Layout::from_sizes_and_strides([(5, 16)]), Layout::from_sizes_and_strides([(2, 80), (2, 4)])]
        ))]
    fn composition(#[case] a: Layout, #[case] b: Layout, #[case] c: Layout) {
        assert_eq!(a.compose(&b), c);
    }
}
