use std::ops::Index;

/// Shape of a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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