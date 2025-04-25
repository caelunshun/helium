use crate::{Device, Tensor, module::Module, raw_tensor::RawTensor};
use ::safetensors::SafeTensorError;
use serde::{Serialize, de::DeserializeOwned};
use std::io;

pub mod safetensors;

/// Implements serialization for modules.
pub trait Recorder {
    fn record_tensor<const D: usize>(
        &mut self,
        name: &str,
        param: &Tensor<D>,
    ) -> Result<(), RecordError> {
        self.record_raw_tensor(name, &param.clone().into_raw())
    }

    fn record_raw_tensor(&mut self, name: &str, tensor: &RawTensor) -> Result<(), RecordError>;

    fn record_config(&mut self, key: &str, value: &impl Serialize) -> Result<(), RecordError>;

    fn record_submodule(&mut self, name: &str, submodule: &impl Module) -> Result<(), RecordError>;
}

/// Implements deserialization for modules.
pub trait ConfigLoader {
    fn load_config<T: DeserializeOwned>(&mut self, key: &str) -> Result<T, RecordError>;

    fn load_submodule<T: Module>(&mut self, name: &str, device: Device) -> Result<T, RecordError>;
}

pub trait TensorLoader {
    fn load_tensor<const D: usize>(
        &mut self,
        name: &str,
        device: Device,
    ) -> Result<Tensor<D>, RecordError> {
        let raw = self.load_raw_tensor(name, device)?;
        if raw.shape().num_dims() != D {
            return Err(RecordError::Other(format!(
                "expected {D} dimensions, but record contained {}",
                raw.shape().num_dims()
            )));
        }
        Ok(Tensor::from_raw(raw))
    }

    fn load_raw_tensor(&mut self, name: &str, device: Device) -> Result<RawTensor, RecordError>;

    fn load_submodule(&mut self, name: &str, module: &mut impl Module) -> Result<(), RecordError>;
}

#[derive(Debug, thiserror::Error)]
pub enum RecordError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("missing field (param, config, or submodule) '{0}'")]
    MissingField(String),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Safetensors(#[from] SafeTensorError),
    #[error("{0}")]
    Other(String),
}
