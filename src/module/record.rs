use crate::{module::Module, Device, Tensor};
use serde::{de::DeserializeOwned, Serialize};
use std::io;

/// Implements serialization for modules.
pub trait Recorder {
    fn record_param<const D: usize>(
        &mut self,
        name: &str,
        param: &Tensor<D>,
    ) -> Result<(), RecordError>;

    fn record_config(&mut self, key: &str, value: &impl Serialize) -> Result<(), RecordError>;

    fn record_submodule(&mut self, name: &str, submodule: &impl Module) -> Result<(), RecordError>;
}

/// Implements deserialization for modules.
pub trait ConfigLoader {
    fn load_config<T: DeserializeOwned>(&mut self, key: &str) -> Result<T, RecordError>;

    fn load_submodule<T: Module>(&mut self, name: &str, device: Device) -> Result<T, RecordError>;
}

pub trait ParamLoader {
    fn load_param<const D: usize>(
        &mut self,
        name: &str,
        device: Device,
    ) -> Result<Tensor<D>, RecordError>;

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
    #[error("{0}")]
    Other(String),
}
