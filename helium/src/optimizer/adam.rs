use crate::{
    Gradients, Module, Param, ParamId, Tensor, module::ParamMutVisitor, optimizer::Optimizer,
    raw_tensor::RawTensor,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Module)]
pub struct Adam {
    first_moments: BTreeMap<ParamId, RawTensor>,
    second_moments: BTreeMap<ParamId, RawTensor>,
    #[module(config)]
    settings: AdamSettings,
    #[module(config)]
    t: u64,
}

impl Adam {
    pub fn new(settings: AdamSettings) -> Self {
        Self {
            settings,
            first_moments: BTreeMap::default(),
            second_moments: BTreeMap::default(),
            t: 1,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, module: &mut impl Module, gradients: &Gradients, learning_rate: f32) {
        struct Visitor<'a> {
            optimizer: &'a mut Adam,
            gradients: &'a Gradients,
            learning_rate: f32,
        }

        impl ParamMutVisitor for Visitor<'_> {
            fn visit_param_mut<const D: usize>(&mut self, param: &mut Param<D>) {
                let old_weights = param.value().clone();
                let gradients = self.gradients.get::<D>(param.id());

                let first_moment = self
                    .optimizer
                    .first_moments
                    .entry(param.id())
                    .or_insert_with(|| {
                        RawTensor::from_constant(0.0f32, gradients.shape(), gradients.device())
                    });
                let second_moment = self
                    .optimizer
                    .second_moments
                    .entry(param.id())
                    .or_insert_with(|| {
                        RawTensor::from_constant(0.0f32, gradients.shape(), gradients.device())
                    });

                *first_moment = first_moment.clone() * self.optimizer.settings.beta1
                    + gradients.clone().into_raw() * (1.0 - self.optimizer.settings.beta1);
                *second_moment = second_moment.clone() * self.optimizer.settings.beta2
                    + gradients.clone().into_raw().pow_scalar(2.0)
                        * (1.0 - self.optimizer.settings.beta2);

                let first_moment = first_moment.clone()
                    / (1.0 - self.optimizer.settings.beta1.powi(self.optimizer.t as i32));
                let second_moment = second_moment.clone()
                    / (1.0 - self.optimizer.settings.beta2.powi(self.optimizer.t as i32));

                let first_moment = Tensor::<D>::from_raw(first_moment.clone());
                let second_moment = Tensor::<D>::from_raw(second_moment.clone());

                let new_weights =
                    old_weights - first_moment * self.learning_rate / (second_moment.sqrt() + 1e-6);
                param.set_value(new_weights);
            }
        }

        module.visit_params_mut(&mut Visitor {
            optimizer: self,
            gradients,
            learning_rate,
        });
        self.t += 1;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamSettings {
    pub beta1: f32,
    pub beta2: f32,
}

impl Default for AdamSettings {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
        }
    }
}
