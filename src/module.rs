use crate::{data_type::DataClassTrait, Param, Tensor};

pub trait Module: Send + Sync {
    /// Call `visitor.visit_param` on all the tensor parameters of the module.
    fn visit_params(&self, visitor: &mut impl ParamVisitor);
    /// Call `visitor.visit_param_mut` on all the tensor parameters of the module.
    fn visit_params_mut(&mut self, visitor: &mut impl ParamMutVisitor);
}

pub trait ParamVisitor {
    fn visit_param<const D: usize>(&mut self, param: &Param<D>);
}

pub trait ParamMutVisitor {
    fn visit_param_mut<const D: usize>(&mut self, param: &mut Param<D>);
}

impl<T: Module> Module for Option<T> {
    fn visit_params(&self, visitor: &mut impl ParamVisitor) {
        if let Some(module) = self {
            module.visit_params(visitor);
        }
    }

    fn visit_params_mut(&mut self, visitor: &mut impl ParamMutVisitor) {
        if let Some(module) = self {
            module.visit_params_mut(visitor);
        }
    }
}

impl<T: Module> Module for Vec<T> {
    fn visit_params(&self, visitor: &mut impl ParamVisitor) {
        for module in self {
            module.visit_params(visitor);
        }
    }

    fn visit_params_mut(&mut self, visitor: &mut impl ParamMutVisitor) {
        for module in self {
            module.visit_params_mut(visitor);
        }
    }
}

impl Module for () {
    fn visit_params(&self, visitor: &mut impl ParamVisitor) {
        let _ = visitor;
    }

    fn visit_params_mut(&mut self, visitor: &mut impl ParamMutVisitor) {
        let _ = visitor;
    }
}

impl<const D: usize> Module for Param<D> {
    fn visit_params(&self, visitor: &mut impl ParamVisitor) {
        visitor.visit_param(self);
    }

    fn visit_params_mut(&mut self, visitor: &mut impl ParamMutVisitor) {
        visitor.visit_param_mut(self);
    }
}

impl<const D: usize, C: DataClassTrait> Module for Tensor<D, C> {
    fn visit_params(&self, visitor: &mut impl ParamVisitor) {
        // Not a parameter
        let _ = visitor;
    }

    fn visit_params_mut(&mut self, visitor: &mut impl ParamMutVisitor) {
        let _ = visitor;
    }
}
