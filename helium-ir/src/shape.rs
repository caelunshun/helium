use serde::{Deserialize, Serialize};
use std::ops::Index;

/// Shape of a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self(dims.into())
    }

    pub fn dim_at(&self, x: i32) -> usize {
        if x >= self.0.len() as i32 || -x > self.0.len() as i32 {
            return 1;
        }

        if x >= 0 {
            self.0[x as usize]
        } else {
            self.0[self.0.len() - ((-x) as usize)]
        }
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn strides(&self) -> impl Iterator<Item = usize> {
        let mut strides = vec![0usize; self.num_dims()];
        let mut stride = 1;
        for (i, dim) in self.dims().iter().enumerate().rev() {
            strides[i] = stride;
            stride *= *dim;
        }
        strides.into_iter()
    }

    pub fn num_dims(&self) -> usize {
        self.0.len()
    }

    pub fn num_elements(&self) -> usize {
        self.0.iter().product()
    }

    pub fn prepend_new_dim(&mut self, size: usize) {
        self.0.insert(0, size);
    }

    pub fn set_dim_size(&mut self, index: usize, size: usize) {
        self.0[index] = size;
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Self::new(value)
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        Self::new(value)
    }
}
