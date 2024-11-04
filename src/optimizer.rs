use helium::{Gradients, Module};

pub mod sgd;

pub trait Optimizer {
    fn step(&mut self, module: &mut impl Module, gradients: &Gradients, learning_rate: f32);
}
