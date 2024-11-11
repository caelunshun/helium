use crate::{
    module::record::{
        safetensors::{SafetensorsLoader, SafetensorsRecorder},
        ConfigLoader, RecordError, Recorder, TensorLoader,
    },
    raw_tensor::RawTensor,
    shape::Shape,
    DataType, Device, Param, ParamId, Tensor,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{array, collections::BTreeMap, fs, path::Path};

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
    fn load_tensors(&mut self, loader: &mut impl TensorLoader) -> Result<(), RecordError>;

    fn save_to_safetensors(&self, path: impl AsRef<Path>) -> Result<(), RecordError> {
        let mut recorder = SafetensorsRecorder::new();
        self.record(&mut recorder)?;
        recorder.save_to_file(path)
    }

    fn load_from_safetensors(path: impl AsRef<Path>, device: Device) -> Result<Self, RecordError> {
        let bytes = fs::read(path)?;
        let mut loader = SafetensorsLoader::new(&bytes)?;
        let mut module = Self::load_config(&mut loader, device)?;
        module.load_tensors(&mut loader)?;
        Ok(module)
    }

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

    fn load_tensors(&mut self, loader: &mut impl TensorLoader) -> Result<(), RecordError> {
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

    fn load_tensors(&mut self, loader: &mut impl TensorLoader) -> Result<(), RecordError> {
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

    fn load_tensors(&mut self, loader: &mut impl TensorLoader) -> Result<(), RecordError> {
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

    fn load_tensors(&mut self, loader: &mut impl TensorLoader) -> Result<(), RecordError> {
        let _ = loader;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorConfig {
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
        recorder.record_config("id", &self.id())?;
        self.value().record(recorder)
    }

    fn load_config(loader: &mut impl ConfigLoader, device: Device) -> Result<Self, RecordError> {
        let id = loader.load_config::<ParamId>("id")?;
        Tensor::<D>::load_config(loader, device).map(|tensor| Self::new_with_id(tensor, id))
    }

    fn load_tensors(&mut self, loader: &mut impl TensorLoader) -> Result<(), RecordError> {
        self.set_value(loader.load_tensor::<D>("value", self.value().device())?);
        Ok(())
    }
}

impl<const D: usize> Module for Tensor<D> {
    fn visit_params(&self, visitor: &mut impl ParamVisitor) {
        self.as_raw().visit_params(visitor)
    }

    fn visit_params_mut(&mut self, visitor: &mut impl ParamMutVisitor) {
        self.as_raw_mut().visit_params_mut(visitor)
    }

    fn record(&self, recorder: &mut impl Recorder) -> Result<(), RecordError> {
        self.as_raw().record(recorder)
    }

    fn load_config(loader: &mut impl ConfigLoader, device: Device) -> Result<Self, RecordError> {
        let raw = RawTensor::load_config(loader, device)?;
        if raw.shape().num_dims() != D {
            return Err(RecordError::Other("shape mismatch".to_owned()));
        }
        Ok(Tensor::from_raw(raw))
    }

    fn load_tensors(&mut self, loader: &mut impl TensorLoader) -> Result<(), RecordError> {
        self.as_raw_mut().load_tensors(loader)
    }
}

impl Module for RawTensor {
    fn visit_params(&self, _visitor: &mut impl ParamVisitor) {}

    fn visit_params_mut(&mut self, _visitor: &mut impl ParamMutVisitor) {}

    fn record(&self, recorder: &mut impl Recorder) -> Result<(), RecordError> {
        recorder.record_config(
            "meta",
            &TensorConfig {
                data_type: self.data_type(),
                shape: self.shape(),
            },
        )?;
        recorder.record_raw_tensor("value", self)?;
        Ok(())
    }

    fn load_config(loader: &mut impl ConfigLoader, device: Device) -> Result<Self, RecordError> {
        let meta: TensorConfig = loader.load_config("meta")?;
        Ok(RawTensor::from_constant(0.0f32, meta.shape.clone(), device))
    }

    fn load_tensors(&mut self, loader: &mut impl TensorLoader) -> Result<(), RecordError> {
        *self = loader.load_raw_tensor("value", self.device())?;
        Ok(())
    }
}

impl<T> Module for Box<T>
where
    T: Module,
{
    fn visit_params(&self, visitor: &mut impl ParamVisitor) {
        (**self).visit_params(visitor)
    }

    fn visit_params_mut(&mut self, visitor: &mut impl ParamMutVisitor) {
        (**self).visit_params_mut(visitor)
    }

    fn record(&self, recorder: &mut impl Recorder) -> Result<(), RecordError> {
        (**self).record(recorder)
    }

    fn load_config(loader: &mut impl ConfigLoader, device: Device) -> Result<Self, RecordError> {
        T::load_config(loader, device).map(Box::new)
    }

    fn load_tensors(&mut self, loader: &mut impl TensorLoader) -> Result<(), RecordError> {
        (**self).load_tensors(loader)
    }
}

impl<K, V> Module for BTreeMap<K, V>
where
    K: Ord + Serialize + DeserializeOwned + Send + Sync,
    V: Module + Send + Sync,
{
    fn visit_params(&self, visitor: &mut impl ParamVisitor) {
        for module in self.values() {
            module.visit_params(visitor);
        }
    }

    fn visit_params_mut(&mut self, visitor: &mut impl ParamMutVisitor) {
        for module in self.values_mut() {
            module.visit_params_mut(visitor);
        }
    }

    fn record(&self, recorder: &mut impl Recorder) -> Result<(), RecordError> {
        let in_order: Vec<(&K, &V)> = self.iter().collect();
        recorder.record_config("keys", &in_order.iter().map(|(k, _)| k).collect::<Vec<_>>())?;
        for (i, (_, module)) in in_order.iter().enumerate() {
            recorder.record_submodule(&i.to_string(), *module)?;
        }
        Ok(())
    }

    fn load_config(loader: &mut impl ConfigLoader, device: Device) -> Result<Self, RecordError> {
        let keys: Vec<K> = loader.load_config("keys")?;
        let mut map = BTreeMap::new();
        for (i, key) in keys.into_iter().enumerate() {
            let module: V = loader.load_submodule(&i.to_string(), device)?;
            map.insert(key, module);
        }
        Ok(map)
    }

    fn load_tensors(&mut self, loader: &mut impl TensorLoader) -> Result<(), RecordError> {
        for (i, module) in self.values_mut().enumerate() {
            loader.load_submodule(&i.to_string(), module)?;
        }
        Ok(())
    }
}
