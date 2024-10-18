//! Somewhat safe wrapper over the cuDNN
//! Graph API.

#![allow(unused)]

use crate::{cuda::error::CudaError, data_type::DataType, shape::Shape};
use cudarc::cudnn::{
    sys::{
        cudnnBackendAttributeName_t::*, cudnnBackendAttributeType_t::*,
        cudnnBackendDescriptorType_t::*, cudnnDataType_t::CUDNN_DATA_FLOAT, *,
    },
    CudnnError,
};
use std::{
    ffi::c_void,
    iter, mem,
    mem::MaybeUninit,
    ptr,
    sync::{atomic::AtomicU64, Arc},
};

pub struct CudnnContext(Arc<ContextInner>);

impl CudnnContext {
    pub fn new() -> Result<Self, CudaError> {
        let mut handle = ptr::null_mut();
        unsafe {
            lib().cudnnCreate(&mut handle).result()?;
        }
        Ok(Self(Arc::new(ContextInner { handle })))
    }
}

struct ContextInner {
    handle: cudnnHandle_t,
}

unsafe impl Send for ContextInner {}
unsafe impl Sync for ContextInner {}

impl Drop for ContextInner {
    fn drop(&mut self) {
        unsafe {
            lib()
                .cudnnDestroy(self.handle)
                .result()
                .expect("failed to destroy cudnn handle");
        }
    }
}

#[repr(transparent)]
pub struct RawDescriptor {
    desc: cudnnBackendDescriptor_t,
}

impl RawDescriptor {
    pub fn new(typ: cudnnBackendDescriptorType_t) -> Result<Self, CudaError> {
        let mut desc = ptr::null_mut();
        unsafe {
            lib()
                .cudnnBackendCreateDescriptor(typ, &mut desc)
                .result()?;
        }
        Ok(Self { desc })
    }

    pub fn into_inner(self) -> cudnnBackendDescriptor_t {
        let desc = self.desc;
        mem::forget(self);
        desc
    }

    pub fn set_attribute<A: Attribute>(
        &mut self,
        name: cudnnBackendAttributeName_t,
        value: A,
    ) -> Result<(), CudaError> {
        self.set_attribute_slice(name, &[value])
    }

    pub fn set_attribute_slice<A: Attribute>(
        &mut self,
        name: cudnnBackendAttributeName_t,
        values: &[A],
    ) -> Result<(), CudaError> {
        unsafe {
            lib().cudnnBackendSetAttribute(
                self.desc,
                name,
                A::attrib_type(),
                values.len().try_into().unwrap(),
                values.as_ptr() as *mut _,
            );
        }
        Ok(())
    }

    pub fn get_attribute<A: Attribute>(
        &self,
        name: cudnnBackendAttributeName_t,
    ) -> Result<A, CudaError> {
        let mut val = MaybeUninit::<A>::uninit();
        let mut element_count = 0;
        unsafe {
            lib()
                .cudnnBackendGetAttribute(
                    self.desc,
                    name,
                    A::attrib_type(),
                    1,
                    &mut element_count,
                    &mut val as *mut _ as *mut _,
                )
                .result()?;
            Ok(val.assume_init())
        }
    }

    pub fn get_attribute_vec<A: Attribute>(
        &self,
        name: cudnnBackendAttributeName_t,
        init: impl Fn() -> Result<MaybeUninit<A>, CudaError>,
    ) -> Result<Vec<A>, CudaError> {
        let mut count = 0;
        unsafe {
            lib().cudnnBackendGetAttribute(
                self.desc,
                name,
                A::attrib_type(),
                0,
                &mut count,
                ptr::null_mut(),
            );
        };

        let mut values = iter::repeat_with(init)
            .take(count.try_into().unwrap())
            .collect::<Result<Vec<MaybeUninit<A>>, CudaError>>()?;
        let mut actual_count = 0;
        unsafe {
            lib()
                .cudnnBackendGetAttribute(
                    self.desc,
                    name,
                    A::attrib_type(),
                    count,
                    &mut actual_count,
                    values.as_mut_ptr() as *mut _,
                )
                .result()?;
        }
        values.truncate(actual_count.try_into().unwrap());
        Ok(values
            .into_iter()
            .map(|val| unsafe { val.assume_init() })
            .collect())
    }

    pub fn finalize(&mut self) -> Result<(), CudaError> {
        unsafe {
            lib().cudnnBackendFinalize(self.desc).result()?;
        }
        Ok(())
    }
}

impl Drop for RawDescriptor {
    fn drop(&mut self) {
        unsafe {
            lib()
                .cudnnBackendDestroyDescriptor(self.desc)
                .result()
                .expect("failed to destroy cuDNN descriptor")
        }
    }
}

unsafe impl Send for RawDescriptor {}
unsafe impl Sync for RawDescriptor {}

pub unsafe trait Attribute {
    fn attrib_type() -> cudnnBackendAttributeType_t;
}

#[repr(transparent)]
struct VoidPtrAttrib(*mut c_void);

unsafe impl Attribute for VoidPtrAttrib {
    fn attrib_type() -> cudnnBackendAttributeType_t {
        CUDNN_TYPE_VOID_PTR
    }
}

unsafe impl Attribute for cudnnHandle_t {
    fn attrib_type() -> cudnnBackendAttributeType_t {
        CUDNN_TYPE_HANDLE
    }
}

unsafe impl Attribute for cudnnDataType_t {
    fn attrib_type() -> cudnnBackendAttributeType_t {
        CUDNN_TYPE_DATA_TYPE
    }
}

unsafe impl Attribute for bool {
    fn attrib_type() -> cudnnBackendAttributeType_t {
        CUDNN_TYPE_BOOLEAN
    }
}

unsafe impl Attribute for u64 {
    fn attrib_type() -> cudnnBackendAttributeType_t {
        CUDNN_TYPE_INT64
    }
}

unsafe impl Attribute for i64 {
    fn attrib_type() -> cudnnBackendAttributeType_t {
        CUDNN_TYPE_INT64
    }
}

unsafe impl Attribute for u32 {
    fn attrib_type() -> cudnnBackendAttributeType_t {
        CUDNN_TYPE_INT32
    }
}

unsafe impl Attribute for i32 {
    fn attrib_type() -> cudnnBackendAttributeType_t {
        CUDNN_TYPE_INT32
    }
}

unsafe impl Attribute for f32 {
    fn attrib_type() -> cudnnBackendAttributeType_t {
        CUDNN_TYPE_FLOAT
    }
}

unsafe impl Attribute for cudnnBackendDescriptor_t {
    fn attrib_type() -> cudnnBackendAttributeType_t {
        CUDNN_TYPE_BACKEND_DESCRIPTOR
    }
}

unsafe impl Attribute for cudnnPointwiseMode_t {
    fn attrib_type() -> cudnnBackendAttributeType_t {
        CUDNN_TYPE_POINTWISE_MODE
    }
}

unsafe impl Attribute for cudnnBackendHeurMode_t {
    fn attrib_type() -> cudnnBackendAttributeType_t {
        CUDNN_TYPE_HEUR_MODE
    }
}

unsafe impl Attribute for u8 {
    fn attrib_type() -> cudnnBackendAttributeType_t {
        CUDNN_TYPE_CHAR
    }
}

unsafe impl Attribute for cudnnReduceTensorOp_t {
    fn attrib_type() -> cudnnBackendAttributeType_t {
        CUDNN_TYPE_REDUCTION_OPERATOR_TYPE
    }
}

pub struct TensorDescriptor(Arc<RawDescriptor>, TensorUid, TensorKind);

impl TensorDescriptor {
    pub fn new(
        kind: TensorKind,
        data_type: DataType,
        shape: &Shape,
    ) -> Result<TensorDescriptor, CudaError> {
        let id = TensorUid::new();

        let mut desc =
            RawDescriptor::new(cudnnBackendDescriptorType_t::CUDNN_BACKEND_TENSOR_DESCRIPTOR)?;
        desc.set_attribute_slice(
            CUDNN_ATTR_TENSOR_DIMENSIONS,
            make_shape_vec(shape).as_slice(),
        )?;
        desc.set_attribute_slice(CUDNN_ATTR_TENSOR_STRIDES, compute_strides(shape).as_slice())?;
        desc.set_attribute(CUDNN_ATTR_TENSOR_UNIQUE_ID, id.0)?;
        desc.set_attribute(CUDNN_ATTR_TENSOR_DATA_TYPE, convert_data_type(data_type))?;
        desc.set_attribute(CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, 16u64)?;

        if kind == TensorKind::Virtual {
            desc.set_attribute(CUDNN_ATTR_TENSOR_IS_VIRTUAL, true)?;
        }

        desc.finalize()?;
        Ok(Self(Arc::new(desc), id, kind))
    }

    pub fn id(&self) -> TensorUid {
        self.1
    }

    pub fn is_virtual(&self) -> bool {
        self.2 == TensorKind::Virtual
    }
}

unsafe impl Send for TensorDescriptor {}
unsafe impl Sync for TensorDescriptor {}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TensorKind {
    Concrete,
    Virtual,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorUid(u64);

impl TensorUid {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

pub unsafe trait OpDescriptor {
    fn raw(&self) -> &RawDescriptor;
}

pub struct MatmulOpDescriptor {
    matmul: Arc<RawDescriptor>,
    op: Arc<RawDescriptor>,
    refs: Vec<Arc<RawDescriptor>>,
}

unsafe impl OpDescriptor for MatmulOpDescriptor {
    fn raw(&self) -> &RawDescriptor {
        &self.op
    }
}

impl MatmulOpDescriptor {
    pub fn new(
        compute_type: DataType,
        matrix_a: &TensorDescriptor,
        matrix_b: &TensorDescriptor,
        matrix_c: &TensorDescriptor,
    ) -> Result<Self, CudaError> {
        let mut matmul = RawDescriptor::new(CUDNN_BACKEND_MATMUL_DESCRIPTOR)?;
        matmul.set_attribute(CUDNN_ATTR_MATMUL_COMP_TYPE, convert_data_type(compute_type))?;
        matmul.finalize()?;

        let mut op = RawDescriptor::new(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)?;
        op.set_attribute(CUDNN_ATTR_OPERATION_MATMUL_ADESC, matrix_a.0.desc)?;
        op.set_attribute(CUDNN_ATTR_OPERATION_MATMUL_BDESC, matrix_b.0.desc)?;
        op.set_attribute(CUDNN_ATTR_OPERATION_MATMUL_CDESC, matrix_c.0.desc)?;
        op.set_attribute(CUDNN_ATTR_OPERATION_MATMUL_DESC, matmul.desc)?;
        op.finalize()?;

        Ok(Self {
            matmul: Arc::new(matmul),
            op: Arc::new(op),
            refs: vec![matrix_a.0.clone(), matrix_b.0.clone(), matrix_c.0.clone()],
        })
    }
}

pub struct PointwiseOpDescriptor {
    pointwise: Arc<RawDescriptor>,
    op: Arc<RawDescriptor>,
    refs: Vec<Arc<RawDescriptor>>,
}

unsafe impl OpDescriptor for PointwiseOpDescriptor {
    fn raw(&self) -> &RawDescriptor {
        &self.op
    }
}

impl PointwiseOpDescriptor {
    pub fn new(
        mode: PointwiseMode,
        precision: DataType,
        input_a: &TensorDescriptor,
        input_b: Option<&TensorDescriptor>,
        output: &TensorDescriptor,
    ) -> Result<Self, CudaError> {
        let mut pointwise = RawDescriptor::new(CUDNN_BACKEND_POINTWISE_DESCRIPTOR)?;
        pointwise.set_attribute(CUDNN_ATTR_POINTWISE_MODE, mode.to_cudnn())?;
        pointwise.set_attribute(CUDNN_ATTR_POINTWISE_MATH_PREC, convert_data_type(precision))?;
        pointwise.finalize()?;

        let mut op = RawDescriptor::new(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)?;
        op.set_attribute(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pointwise.desc)?;
        op.set_attribute(CUDNN_ATTR_OPERATION_POINTWISE_XDESC, input_a.0.desc)?;
        if let Some(input_b) = input_b {
            op.set_attribute(CUDNN_ATTR_OPERATION_POINTWISE_BDESC, input_b.0.desc)?;
        }
        op.set_attribute(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, output.0.desc)?;

        op.finalize()?;

        let mut refs = vec![input_a.0.clone(), output.0.clone()];
        if let Some(input_b) = input_b {
            refs.push(input_b.0.clone());
        }

        Ok(Self {
            pointwise: Arc::new(pointwise),
            op: Arc::new(op),
            refs,
        })
    }
}

#[derive(Debug, Clone)]
pub enum PointwiseMode {
    Identity,
    Add,
    Div,
    Max,
    Min,
    Mod,
    Mul,
    Pow,
    Sub,
    Abs,
    Ceil,
    Cos,
    Exp,
    Floor,
    Log,
    Neg,
    Rsqrt,
    Sin,
    Sqrt,
    Tan,
    Erf,
    Recip,
    Tanh,
    Sigmoid,
    Relu,
}

impl PointwiseMode {
    fn to_cudnn(&self) -> cudnnPointwiseMode_t {
        use cudnnPointwiseMode_t::*;
        match self {
            PointwiseMode::Identity => CUDNN_POINTWISE_IDENTITY,
            PointwiseMode::Add => CUDNN_POINTWISE_ADD,
            PointwiseMode::Div => CUDNN_POINTWISE_DIV,
            PointwiseMode::Max => CUDNN_POINTWISE_MAX,
            PointwiseMode::Min => CUDNN_POINTWISE_MIN,
            PointwiseMode::Mod => CUDNN_POINTWISE_MOD,
            PointwiseMode::Mul => CUDNN_POINTWISE_MUL,
            PointwiseMode::Pow => CUDNN_POINTWISE_POW,
            PointwiseMode::Sub => CUDNN_POINTWISE_SUB,
            PointwiseMode::Abs => CUDNN_POINTWISE_ABS,
            PointwiseMode::Ceil => CUDNN_POINTWISE_CEIL,
            PointwiseMode::Cos => CUDNN_POINTWISE_COS,
            PointwiseMode::Exp => CUDNN_POINTWISE_EXP,
            PointwiseMode::Floor => CUDNN_POINTWISE_FLOOR,
            PointwiseMode::Log => CUDNN_POINTWISE_LOG,
            PointwiseMode::Neg => CUDNN_POINTWISE_NEG,
            PointwiseMode::Rsqrt => CUDNN_POINTWISE_RSQRT,
            PointwiseMode::Sin => CUDNN_POINTWISE_SIN,
            PointwiseMode::Sqrt => CUDNN_POINTWISE_SQRT,
            PointwiseMode::Tan => CUDNN_POINTWISE_TAN,
            PointwiseMode::Erf => CUDNN_POINTWISE_ERF,
            PointwiseMode::Recip => CUDNN_POINTWISE_RECIPROCAL,
            PointwiseMode::Tanh => CUDNN_POINTWISE_TANH_FWD,
            PointwiseMode::Sigmoid => CUDNN_POINTWISE_SIGMOID_FWD,
            PointwiseMode::Relu => CUDNN_POINTWISE_RELU_FWD,
        }
    }
}

pub struct ReductionOpDescriptor {
    reduction: Arc<RawDescriptor>,
    op: Arc<RawDescriptor>,
    refs: Vec<Arc<dyn Send + Sync>>,
}

unsafe impl OpDescriptor for ReductionOpDescriptor {
    fn raw(&self) -> &RawDescriptor {
        &self.op
    }
}

impl ReductionOpDescriptor {
    pub fn new(
        mode: ReductionMode,
        input: &TensorDescriptor,
        output: &TensorDescriptor,
    ) -> Result<Self, CudaError> {
        let mut reduction = RawDescriptor::new(CUDNN_BACKEND_REDUCTION_DESCRIPTOR)?;
        reduction.set_attribute(CUDNN_ATTR_REDUCTION_OPERATOR, mode.to_cudnn())?;
        reduction.set_attribute(CUDNN_ATTR_REDUCTION_COMP_TYPE, CUDNN_DATA_FLOAT)?;
        reduction.finalize()?;

        let mut op = RawDescriptor::new(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)?;
        op.set_attribute(CUDNN_ATTR_OPERATION_REDUCTION_DESC, reduction.desc)?;
        op.set_attribute(CUDNN_ATTR_OPERATION_REDUCTION_XDESC, input.0.desc)?;
        op.set_attribute(CUDNN_ATTR_OPERATION_REDUCTION_YDESC, output.0.desc)?;
        op.finalize()?;

        Ok(Self {
            reduction: Arc::new(reduction),
            op: Arc::new(op),
            refs: vec![input.0.clone(), output.0.clone()],
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReductionMode {
    Add,
    Mul,
    Min,
    Max,
    Mean,
}

impl ReductionMode {
    fn to_cudnn(self) -> cudnnReduceTensorOp_t {
        use cudnnReduceTensorOp_t::*;
        match self {
            ReductionMode::Add => CUDNN_REDUCE_TENSOR_ADD,
            ReductionMode::Mul => CUDNN_REDUCE_TENSOR_MUL,
            ReductionMode::Min => CUDNN_REDUCE_TENSOR_MIN,
            ReductionMode::Max => CUDNN_REDUCE_TENSOR_MAX,
            ReductionMode::Mean => CUDNN_REDUCE_TENSOR_AVG,
        }
    }
}

pub struct OperationGraph {
    raw: Arc<RawDescriptor>,
    refs: Vec<Arc<dyn Send + Sync>>,
    cx: Arc<ContextInner>,
}

impl OperationGraph {
    pub fn builder() -> OperationGraphBuilder {
        OperationGraphBuilder::default()
    }
}

#[derive(Default)]
pub struct OperationGraphBuilder {
    ops: Vec<cudnnBackendDescriptor_t>,
    refs: Vec<Arc<dyn Send + Sync>>,
}

impl OperationGraphBuilder {
    pub fn add_op(&mut self, op: impl OpDescriptor + Send + Sync + 'static) -> &mut Self {
        self.ops.push(op.raw().desc);
        self.refs.push(Arc::new(op));
        self
    }

    pub fn with_op(mut self, op: impl OpDescriptor + Send + Sync + 'static) -> Self {
        self.add_op(op);
        self
    }

    pub fn build(self, cx: &CudnnContext) -> Result<OperationGraph, CudaError> {
        let mut desc = RawDescriptor::new(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)?;
        desc.set_attribute(CUDNN_ATTR_OPERATIONGRAPH_HANDLE, cx.0.handle)?;
        desc.set_attribute_slice(CUDNN_ATTR_OPERATIONGRAPH_OPS, &self.ops)?;
        desc.finalize()?;

        Ok(OperationGraph {
            raw: Arc::new(desc),
            refs: self.refs,
            cx: cx.0.clone(),
        })
    }
}

pub struct Engine {
    engine: Arc<RawDescriptor>,
    config: Arc<RawDescriptor>,
    plan: Arc<RawDescriptor>,
    refs: Vec<Arc<dyn Send + Sync>>,
    cx: Arc<ContextInner>,
}

impl Engine {
    pub fn choose_with_heuristic(graph: &OperationGraph) -> Result<Self, CudaError> {
        let mut heuristic = RawDescriptor::new(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR)?;
        heuristic.set_attribute(CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, graph.raw.desc)?;
        heuristic.set_attribute(
            CUDNN_ATTR_ENGINEHEUR_MODE,
            cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_A,
        )?;
        heuristic.finalize()?;

        let configs = heuristic.get_attribute_vec::<cudnnBackendDescriptor_t>(
            CUDNN_ATTR_ENGINEHEUR_RESULTS,
            || {
                RawDescriptor::new(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR)
                    .map(RawDescriptor::into_inner)
                    .map(MaybeUninit::new)
            },
        )?;
        if configs.is_empty() {
            return Err(CudaError::Other("no engine configs available".to_string()));
        }
        let config = configs[0];
        let config = RawDescriptor { desc: config };

        let engine = config
            .get_attribute_vec::<cudnnBackendDescriptor_t>(CUDNN_ATTR_ENGINECFG_ENGINE, || {
                RawDescriptor::new(CUDNN_BACKEND_ENGINE_DESCRIPTOR)
                    .map(RawDescriptor::into_inner)
                    .map(MaybeUninit::new)
            })?
            .remove(0);
        let engine = RawDescriptor { desc: engine };

        let mut plan = RawDescriptor::new(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR)?;
        plan.set_attribute(CUDNN_ATTR_EXECUTION_PLAN_HANDLE, graph.cx.handle)?;
        plan.set_attribute(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, config.desc)?;
        plan.finalize()?;

        let mut refs = graph.refs.clone();
        refs.push(graph.raw.clone());

        let json = plan.get_attribute_vec(CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION, || {
            Ok(MaybeUninit::<u8>::uninit())
        })?;
        //println!("{}", std::str::from_utf8(&json).unwrap());

        Ok(Self {
            engine: Arc::new(engine),
            config: Arc::new(config),
            plan: Arc::new(plan),
            refs,
            cx: graph.cx.clone(),
        })
    }

    pub fn workspace_size(&self) -> Result<usize, CudaError> {
        Ok(self
            .plan
            .get_attribute::<u64>(CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE)?
            .try_into()
            .unwrap())
    }

    pub unsafe fn execute(
        &self,
        varpack: &VariantPack,
        stream: cudaStream_t,
    ) -> Result<(), CudaError> {
        unsafe {
            lib().cudnnSetStream(self.cx.handle, stream).result()?;
            lib()
                .cudnnBackendExecute(self.cx.handle, self.plan.desc, varpack.raw.desc)
                .result()?;
        }

        Ok(())
    }
}

pub struct VariantPack {
    raw: Arc<RawDescriptor>,
}

impl VariantPack {
    pub fn builder() -> VariantPackBuilder {
        VariantPackBuilder::default()
    }
}

#[derive(Default)]
pub struct VariantPackBuilder {
    device_ptrs: Vec<VoidPtrAttrib>,
    uids: Vec<u64>,
}

impl VariantPackBuilder {
    pub fn add_tensor(&mut self, uid: TensorUid, device_ptr: *mut c_void) -> &mut Self {
        self.device_ptrs.push(VoidPtrAttrib(device_ptr));
        self.uids.push(uid.0);
        self
    }

    pub fn with_tensor(mut self, uid: TensorUid, device_ptr: *mut c_void) -> Self {
        self.add_tensor(uid, device_ptr);
        self
    }

    pub unsafe fn build(self, workspace: *mut c_void) -> Result<VariantPack, CudaError> {
        let mut desc = RawDescriptor::new(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR)?;
        desc.set_attribute_slice(CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, &self.device_ptrs)?;
        desc.set_attribute_slice(CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, &self.uids)?;
        desc.set_attribute(CUDNN_ATTR_VARIANT_PACK_WORKSPACE, VoidPtrAttrib(workspace))?;
        desc.finalize()?;
        Ok(VariantPack {
            raw: Arc::new(desc),
        })
    }
}

trait ToResult {
    fn result(self) -> Result<(), CudaError>;
}

impl ToResult for cudnnStatus_t {
    fn result(self) -> Result<(), CudaError> {
        match self {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            _ => Err(CudaError::Cudnn(CudnnError(self))),
        }
    }
}

fn convert_data_type(dtype: DataType) -> cudnnDataType_t {
    match dtype {
        DataType::F16 => cudnnDataType_t::CUDNN_DATA_HALF,
        DataType::Bf16 => cudnnDataType_t::CUDNN_DATA_BFLOAT16,
        DataType::F32 => cudnnDataType_t::CUDNN_DATA_FLOAT,
        DataType::U32 => cudnnDataType_t::CUDNN_DATA_INT32,
        DataType::Bool => cudnnDataType_t::CUDNN_DATA_BOOLEAN,
    }
}

fn make_shape_vec(shape: &Shape) -> Vec<u64> {
    shape.dims().iter().copied().map(|x| x as u64).collect()
}

fn compute_strides(shape: &Shape) -> Vec<u64> {
    let mut strides = vec![0u64; shape.num_dims()];
    let mut product = 1;
    for (&dim, stride) in shape.dims().iter().rev().zip(&mut strides) {
        *stride = product;
        product *= dim as u64;
    }
    strides.reverse();
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::context::CudaContext;
    use approx::assert_abs_diff_eq;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use faer::Mat;

    #[test]
    fn test_compute_strides() {
        assert_eq!(
            compute_strides(&Shape::new([1, 2, 3, 4])),
            vec![24, 12, 4, 1],
        );
    }

    #[test]
    fn execute_matmul() {
        let cuda = CudaContext::new(0).unwrap();

        let cx = CudnnContext::new().unwrap();

        let size = 64;

        let desc_a = TensorDescriptor::new(
            TensorKind::Concrete,
            DataType::F32,
            &Shape::new([1, size, size]),
        )
        .unwrap();
        let desc_b = TensorDescriptor::new(
            TensorKind::Concrete,
            DataType::F32,
            &Shape::new([1, size, size]),
        )
        .unwrap();
        let desc_imm = TensorDescriptor::new(
            TensorKind::Virtual,
            DataType::F32,
            &Shape::new([1, size, size]),
        )
        .unwrap();
        let desc_out = TensorDescriptor::new(
            TensorKind::Concrete,
            DataType::F32,
            &Shape::new([1, size, size]),
        )
        .unwrap();
        let desc_reduced =
            TensorDescriptor::new(TensorKind::Concrete, DataType::F32, &Shape::new([1, 1, 1]))
                .unwrap();

        let op_matmul =
            MatmulOpDescriptor::new(DataType::F32, &desc_a, &desc_b, &desc_imm).unwrap();
        let op_cos = PointwiseOpDescriptor::new(
            PointwiseMode::Cos,
            DataType::F32,
            &desc_imm,
            None,
            &desc_out,
        )
        .unwrap();
        let op_reduce =
            ReductionOpDescriptor::new(ReductionMode::Add, &desc_imm, &desc_reduced).unwrap();

        let graph = OperationGraph::builder()
            .with_op(op_matmul)
            .with_op(op_cos)
            .with_op(op_reduce)
            .build(&cx)
            .unwrap();

        let engine = Engine::choose_with_heuristic(&graph).unwrap();

        let mat_a: Vec<f32> = (0..size * size).map(|_| rand::random()).collect();
        let mat_b: Vec<f32> = (0..size * size).map(|_| rand::random()).collect();

        let mut dev_a = cuda.device().htod_copy(mat_a.to_vec()).unwrap();
        let mut dev_b = cuda.device().htod_copy(mat_b.to_vec()).unwrap();
        let mut dev_c = cuda.device().alloc_zeros::<f32>(size * size).unwrap();
        let mut dev_reduced = cuda.device().alloc_zeros::<f32>(1).unwrap();

        let workspace = cuda
            .device()
            .alloc_zeros::<u8>(engine.workspace_size().unwrap())
            .unwrap();

        unsafe {
            let varpack = VariantPack::builder()
                .with_tensor(desc_a.id(), *dev_a.device_ptr_mut() as *mut c_void)
                .with_tensor(desc_b.id(), *dev_b.device_ptr_mut() as *mut c_void)
                .with_tensor(desc_out.id(), *dev_c.device_ptr_mut() as *mut c_void)
                .with_tensor(
                    desc_reduced.id(),
                    *dev_reduced.device_ptr_mut() as *mut c_void,
                )
                .build(*workspace.device_ptr() as *mut c_void)
                .unwrap();
            engine.execute(&varpack, ptr::null_mut()).unwrap();
        }

        let result = cuda.device().sync_reclaim(dev_c).unwrap();

        let mat_a = vec2mat(&mat_a);
        let mat_b = vec2mat(&mat_b);
        let mut expected = mat2vec(&(mat_a * mat_b));

        let expected_sum = expected.iter().copied().sum::<f32>();

        expected.iter_mut().for_each(|x| *x = x.cos());

        // cuDNN uses tensorfloat32 precision for matmul (13 fewer bits
        // in the mantissa than f32), so we need to do the comparison with a high epsilon.
        assert_abs_diff_eq!(result.as_slice(), expected.as_slice(), epsilon = 1e-2);

        let actual_sum = cuda.device().sync_reclaim(dev_reduced).unwrap()[0];
        assert_abs_diff_eq!(actual_sum, expected_sum, epsilon = 1.0);
    }

    fn vec2mat(vec: &[f32]) -> Mat<f32> {
        let size = (vec.len() as f64).sqrt() as usize;
        let mut mat: Mat<f32> = Mat::zeros(size, size);
        mat.col_iter_mut()
            .flat_map(|col| col.try_as_slice_mut().unwrap())
            .zip(vec)
            .for_each(|(mat, x): (&mut f32, &f32)| *mat = *x);
        mat.transpose().to_owned()
    }

    fn mat2vec(mat: &Mat<f32>) -> Vec<f32> {
        mat.transpose()
            .to_owned()
            .col_iter()
            .flat_map(|col| col.try_as_slice().unwrap())
            .copied()
            .collect()
    }
}
