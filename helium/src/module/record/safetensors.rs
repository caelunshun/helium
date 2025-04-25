use crate::{
    DataType, Device, Module,
    data_type::DataSlice,
    module::record::{ConfigLoader, RecordError, Recorder, TensorLoader},
    raw_tensor::RawTensor,
    shape::Shape,
};
use bumpalo::Bump;
use pollster::FutureExt;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::{collections::HashMap, path::Path};

/// Serializes a model to a `safetensors` file.
pub struct SafetensorsRecorder {
    tensors: Vec<RawTensor>,
    metadata: serde_json::Map<String, serde_json::Value>,
    current_path: Vec<String>,
}

impl Default for SafetensorsRecorder {
    fn default() -> Self {
        Self {
            tensors: Vec::new(),
            metadata: serde_json::Map::new(),
            current_path: Vec::new(),
        }
    }
}

impl SafetensorsRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn save_to_file(self, path: impl AsRef<Path>) -> Result<(), RecordError> {
        let bump = Bump::new();

        let tensors = self
            .tensors
            .into_iter()
            .enumerate()
            .map(|(i, tensor)| {
                let data = tensor.clone().into_vec().block_on();
                // Lifetime hack to keep DataVecs alive, since TensorView requires
                // a borrowed slice.
                let data = bump.alloc(data);

                let id = format!("_{i}");
                (
                    id,
                    TensorView::new(
                        helium_dtype_to_safetensors(tensor.data_type()),
                        tensor.shape().dims().to_vec(),
                        data.as_bytes(),
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();

        let mut metadata_map = HashMap::new();
        metadata_map.insert("helium".to_owned(), serde_json::to_string(&self.metadata)?);

        safetensors::serialize_to_file(tensors, &Some(metadata_map), path.as_ref())?;

        Ok(())
    }

    fn get_current_metadata(&mut self) -> &mut serde_json::Map<String, serde_json::Value> {
        let mut current = &mut self.metadata;
        for key in &self.current_path {
            current = current.get_mut(key).unwrap().as_object_mut().unwrap();
        }
        current
    }
}

impl Recorder for SafetensorsRecorder {
    fn record_raw_tensor(&mut self, name: &str, param: &RawTensor) -> Result<(), RecordError> {
        let tensor_id = format!("_{}", self.tensors.len());
        self.tensors.push(param.clone());
        self.get_current_metadata()
            .insert(name.to_owned(), serde_json::Value::String(tensor_id));
        Ok(())
    }

    fn record_config(&mut self, key: &str, value: &impl Serialize) -> Result<(), RecordError> {
        self.get_current_metadata()
            .insert(key.to_string(), serde_json::to_value(value)?);
        Ok(())
    }

    fn record_submodule(&mut self, name: &str, submodule: &impl Module) -> Result<(), RecordError> {
        self.get_current_metadata()
            .insert(name.to_owned(), Value::Object(Default::default()));
        self.current_path.push(name.to_owned());
        submodule.record(self)?;
        self.current_path.pop();
        Ok(())
    }
}

pub struct SafetensorsLoader<'a> {
    tensors: SafeTensors<'a>,
    metadata: serde_json::Map<String, serde_json::Value>,
    current_path: Vec<String>,
}

impl<'a> SafetensorsLoader<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self, RecordError> {
        let (_, metadata) = SafeTensors::read_metadata(bytes)?;
        let tensors = SafeTensors::deserialize(bytes)?;

        let metadata = serde_json::from_str(
            metadata
                .metadata()
                .as_ref()
                .ok_or_else(|| RecordError::Other("missing metadata".to_owned()))?
                .get("helium")
                .ok_or_else(|| RecordError::Other("missing helium metadata key".to_owned()))?,
        )?;

        Ok(Self {
            tensors,
            metadata,
            current_path: Vec::new(),
        })
    }

    fn get_current_metadata(&self) -> &serde_json::Map<String, serde_json::Value> {
        let mut current = &self.metadata;
        for key in &self.current_path {
            current = current[key].as_object().unwrap();
        }
        current
    }
}

impl TensorLoader for SafetensorsLoader<'_> {
    fn load_raw_tensor(&mut self, name: &str, device: Device) -> Result<RawTensor, RecordError> {
        let tensor_name = self
            .get_current_metadata()
            .get(name)
            .ok_or_else(|| RecordError::Other(format!("missing tensor '{name}'")))?
            .as_str()
            .ok_or_else(|| RecordError::Other(format!("expected string for key '{name}'")))?;
        let tensor_view = self.tensors.tensor(tensor_name)?;

        let dtype = safetensors_dtype_to_helium(tensor_view.dtype())?;
        let data = DataSlice::from_bytes(dtype, tensor_view.data()).map_err(|e| {
            RecordError::Other(format!(
                "potentially misaligned data in safetensors buffer: {e}"
            ))
        })?;
        let tensor = RawTensor::from_slice(data, Shape::new(tensor_view.shape()), device);

        Ok(tensor)
    }

    fn load_submodule(&mut self, name: &str, module: &mut impl Module) -> Result<(), RecordError> {
        self.current_path.push(name.to_owned());
        module.load_tensors(self)?;
        self.current_path.pop();
        Ok(())
    }
}

impl ConfigLoader for SafetensorsLoader<'_> {
    fn load_config<T: DeserializeOwned>(&mut self, key: &str) -> Result<T, RecordError> {
        let value = self
            .get_current_metadata()
            .get(key)
            .ok_or_else(|| RecordError::Other(format!("missing config key '{key}'")))?;
        serde_json::from_value(value.clone()).map_err(RecordError::Json)
    }

    fn load_submodule<T: Module>(&mut self, name: &str, device: Device) -> Result<T, RecordError> {
        self.current_path.push(name.to_owned());
        let module = T::load_config(self, device)?;
        self.current_path.pop();
        Ok(module)
    }
}

fn helium_dtype_to_safetensors(dt: DataType) -> Dtype {
    match dt {
        DataType::F16 => Dtype::F16,
        DataType::Bf16 => Dtype::BF16,
        DataType::F32 => Dtype::F32,
        DataType::U32 => Dtype::U32,
        DataType::Bool => Dtype::BOOL,
    }
}

fn safetensors_dtype_to_helium(dt: Dtype) -> Result<DataType, RecordError> {
    match dt {
        Dtype::F16 => Ok(DataType::F16),
        Dtype::BF16 => Ok(DataType::Bf16),
        Dtype::F32 => Ok(DataType::F32),
        Dtype::U32 => Ok(DataType::U32),
        Dtype::BOOL => Ok(DataType::Bool),
        dt => Err(RecordError::Other(format!(
            "unsupported safetensors data type {dt:?}"
        ))),
    }
}
