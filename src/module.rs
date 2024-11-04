use crate::{
    module::record::{ConfigLoader, ParamLoader, RecordError, Recorder},
    shape::Shape,
    DataType, Device, Param, Tensor,
};
use serde::{Deserialize, Serialize};
use std::array;

pub mod record;

pub trait Module: Sized + Send + Sync {
    /// Call `visitor.visit_param` on all the tensor parameters of the module.
    fn visit_params(&self, visitor: &mut impl ParamVisitor);
    /// Call `visitor.visit_param_mut` on all the tensor parameters of the module.
    fn visit_params_mut(&mut self, visitor: &mut impl ParamMutVisitor);

    /// Serializes the module config and parameters to a recorder.
    fn record(&self, recorder: &mut impl Recorder) -> Result<(), RecordError>;

    /// Loads the module configuration, setting parameters
    /// to an initial value (e.g. zero).
    fn load_config(loader: &mut impl ConfigLoader, device: Device) -> Result<Self, RecordError>;
    /// Loads the module parameters.
    fn load_params(&mut self, loader: &mut impl ParamLoader) -> Result<(), RecordError>;

    /// Starts evaluating the parameter values on the device.
    fn async_start_eval(&self) {
        struct Visitor;

        impl ParamVisitor for Visitor {
            fn visit_param<const D: usize>(&mut self, param: &Param<D>) {
                param.value().async_start_eval();
            }
        }

        self.visit_params(&mut Visitor);
    }
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

    fn record(&self, recorder: &mut impl Recorder) -> Result<(), RecordError> {
        recorder.record_config("present", &self.is_some())?;
        if let Some(module) = self {
            recorder.record_submodule("value", module)?;
        }
        Ok(())
    }

    fn load_config(loader: &mut impl ConfigLoader, device: Device) -> Result<Self, RecordError> {
        let present: bool = loader.load_config("present")?;
        if present {
            Ok(Some(loader.load_submodule("value", device)?))
        } else {
            Ok(None)
        }
    }

    fn load_params(&mut self, loader: &mut impl ParamLoader) -> Result<(), RecordError> {
        if let Some(module) = self {
            loader.load_submodule("value", module)?;
        }
        Ok(())
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

    fn record(&self, recorder: &mut impl Recorder) -> Result<(), RecordError> {
        recorder.record_config("len", &self.len())?;
        for (i, module) in self.iter().enumerate() {
            recorder.record_submodule(&i.to_string(), module)?;
        }
        Ok(())
    }

    fn load_config(loader: &mut impl ConfigLoader, device: Device) -> Result<Self, RecordError> {
        let len: usize = loader.load_config("len")?;
        let mut modules = Vec::new();
        for i in 0..len {
            modules.push(loader.load_submodule(&i.to_string(), device)?);
        }
        Ok(modules)
    }

    fn load_params(&mut self, loader: &mut impl ParamLoader) -> Result<(), RecordError> {
        for (i, module) in self.iter_mut().enumerate() {
            loader.load_submodule(&i.to_string(), module)?;
        }
        Ok(())
    }
}

impl<const N: usize, T: Module> Module for [T; N] {
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

    fn record(&self, recorder: &mut impl Recorder) -> Result<(), RecordError> {
        for (i, module) in self.iter().enumerate() {
            recorder.record_submodule(&i.to_string(), module)?;
        }
        Ok(())
    }

    fn load_config(loader: &mut impl ConfigLoader, device: Device) -> Result<Self, RecordError> {
        let mut modules = array::from_fn(|_| Option::<T>::None);
        for (i, module) in modules.iter_mut().enumerate() {
            *module = Some(loader.load_submodule(&i.to_string(), device)?);
        }
        Ok(modules.map(|opt| opt.unwrap()))
    }

    fn load_params(&mut self, loader: &mut impl ParamLoader) -> Result<(), RecordError> {
        for (i, module) in self.iter_mut().enumerate() {
            loader.load_submodule(&i.to_string(), module)?;
        }
        Ok(())
    }
}

impl Module for () {
    fn visit_params(&self, visitor: &mut impl ParamVisitor) {
        let _ = visitor;
    }

    fn visit_params_mut(&mut self, visitor: &mut impl ParamMutVisitor) {
        let _ = visitor;
    }

    fn record(&self, recorder: &mut impl Recorder) -> Result<(), RecordError> {
        let _ = recorder;
        Ok(())
    }

    fn load_config(loader: &mut impl ConfigLoader, device: Device) -> Result<Self, RecordError> {
        let _ = (loader, device);
        Ok(())
    }

    fn load_params(&mut self, loader: &mut impl ParamLoader) -> Result<(), RecordError> {
        let _ = loader;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParamMetadata {
    shape: Shape,
    data_type: DataType,
}

impl<const D: usize> Module for Param<D> {
    fn visit_params(&self, visitor: &mut impl ParamVisitor) {
        visitor.visit_param(self);
    }

    fn visit_params_mut(&mut self, visitor: &mut impl ParamMutVisitor) {
        visitor.visit_param_mut(self);
    }

    fn record(&self, recorder: &mut impl Recorder) -> Result<(), RecordError> {
        recorder.record_config(
            "meta",
            &ParamMetadata {
                data_type: self.value().data_type(),
                shape: self.value().shape().into(),
            },
        )?;
        recorder.record_param("value", self.value())?;
        Ok(())
    }

    fn load_config(loader: &mut impl ConfigLoader, device: Device) -> Result<Self, RecordError> {
        let meta: ParamMetadata = loader.load_config("meta")?;
        Ok(Self::new(Tensor::<D>::zeros(
            meta.shape
                .dims()
                .try_into()
                .map_err(|_| RecordError::Other("shape mismatch".to_owned()))?,
            meta.data_type,
            device,
        )))
    }

    fn load_params(&mut self, loader: &mut impl ParamLoader) -> Result<(), RecordError> {
        let device = self.value().device();
        self.set_value(loader.load_param("value", device)?);
        Ok(())
    }
}

impl<const D: usize> Module for Tensor<D> {
    fn visit_params(&self, _visitor: &mut impl ParamVisitor) {}

    fn visit_params_mut(&mut self, _visitor: &mut impl ParamMutVisitor) {}

    fn record(&self, recorder: &mut impl Recorder) -> Result<(), RecordError> {
        recorder.record_config(
            "meta",
            &ParamMetadata {
                data_type: self.data_type(),
                shape: self.shape().into(),
            },
        )?;
        recorder.record_param("value", self)?;
        Ok(())
    }

    fn load_config(loader: &mut impl ConfigLoader, device: Device) -> Result<Self, RecordError> {
        let meta: ParamMetadata = loader.load_config("meta")?;
        Ok(Tensor::<D>::zeros(
            meta.shape
                .dims()
                .try_into()
                .map_err(|_| RecordError::Other("shape mismatch".to_owned()))?,
            meta.data_type,
            device,
        ))
    }

    fn load_params(&mut self, loader: &mut impl ParamLoader) -> Result<(), RecordError> {
        *self = loader.load_param("value", self.device())?;
        Ok(())
    }
}
