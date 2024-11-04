use crate::{
    module::ParamMutVisitor, raw_tensor::RawTensor, DataType, Gradients, Module, Param, ParamId,
    Tensor,
};
use ahash::AHashMap;
use helium::optimizer::Optimizer;

/// Stochastic gradient descent with optional classical
/// momentum.
#[derive(Default)]
pub struct Sgd {
    gradient_averages: GradientAverages,
    momentum: Option<f32>,
}

impl Sgd {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_momentum(momentum: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&momentum),
            "momentum must be in [0, 1]"
        );
        Self {
            momentum: Some(momentum),
            ..Default::default()
        }
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, module: &mut impl Module, gradients: &Gradients, learning_rate: f32) {
        struct Visitor<'a> {
            gradients: &'a Gradients,
            gradient_averages: &'a mut GradientAverages,
            momentum: Option<f32>,
            learning_rate: f32,
        }

        impl ParamMutVisitor for Visitor<'_> {
            fn visit_param_mut<const D: usize>(&mut self, param: &mut Param<D>) {
                let weights = param.value().clone();
                let gradient = self.gradients.get::<D>(param.id());
                let updated_weights = match self.momentum {
                    None => weights - gradient * self.learning_rate,
                    Some(momentum) => {
                        let updated_average =
                            match self.gradient_averages.values.remove(&param.id()) {
                                Some(average) => {
                                    gradient * (1.0 - momentum)
                                        + Tensor::<D>::from_raw(average) * momentum
                                }
                                None => gradient,
                            };
                        self.gradient_averages.values.insert(
                            param.id(),
                            updated_average.to_data_type(DataType::F32).into_raw(),
                        );
                        weights - updated_average * self.learning_rate
                    }
                };
                param.set_value(updated_weights);
            }
        }

        module.visit_params_mut(&mut Visitor {
            gradients,
            gradient_averages: &mut self.gradient_averages,
            momentum: self.momentum,
            learning_rate,
        });
    }
}

#[derive(Default)]
struct GradientAverages {
    values: AHashMap<ParamId, RawTensor>,
}
