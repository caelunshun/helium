use helium::{Gradients, Module};

pub mod adam;
pub mod sgd;

pub use adam::{Adam, AdamSettings};
pub use sgd::Sgd;

pub trait Optimizer {
    fn step(&mut self, module: &mut impl Module, gradients: &Gradients, learning_rate: f32);
}
