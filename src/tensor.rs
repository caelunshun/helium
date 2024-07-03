use crate::{
    data_type::DataType,
    device::Device,
    opgraph::{
        op::{BinaryPointwise, BinaryPointwiseOp, Op, UnaryPointwise, UnaryPointwiseOp},
        Descriptor, NodeId, OpGraph,
    },
};
use parking_lot::Mutex;
use slotmap::{Key, SecondaryMap};
use std::{
    ops::{Add, Mul, Neg, Sub},
    sync::{Arc, Weak},
};

/// Represents a tensor of dimension `D`.
///
/// Tensor data is immutable. Tensors can be cheaply
/// cloned like an `Arc`.
#[derive(Clone)]
pub struct Tensor<const D: usize> {
    inner: Arc<Mutex<TensorInner>>,
}

impl<const D: usize> Tensor<D> {
    fn from_op(cx_arc: &OpGraphContextHandle, op: Op) -> Self {
        let mut cx = cx_arc.lock();
        let inner = Arc::new(Mutex::new(TensorInner {
            data: Data::Virtual(VirtualData {
                graph: Arc::clone(cx_arc),
                node: NodeId::null(),
                shape: op.output_shape(|node| {
                    cx.node_to_tensor[node]
                        .upgrade()
                        .unwrap()
                        .lock()
                        .shape()
                        .to_vec()
                }),
                data_type: op
                    .output_descriptor(|node| cx.op_graph.get(node).descriptor())
                    .data_type,
            }),
        }));

        let node_id = cx.new_op(op, &inner);

        let mut guard = inner.lock();
        let Data::Virtual(data) = &mut guard.data else {
            unreachable!()
        };
        data.node = node_id;
        drop(guard);

        Self { inner }
    }

    /// If the tensor is not already attached to a graph,
    /// we create a new graph. Then, we return the graph
    /// and this tensor's node ID in the graph.
    fn make_graph(&self) -> (OpGraphContextHandle, NodeId) {
        let guard = self.inner.lock();
        match &guard.data {
            Data::Virtual(virt) => (virt.graph.clone(), virt.node),
            Data::Concrete(_) => {
                let cx = OpGraphContextHandle::default();
                let node =
                    cx.lock()
                        .new_input(guard.shape().len() as u32, guard.data_type(), &self.inner);
                (cx, node)
            }
        }
    }

    /// Moves the tensor onto the given graph. If the tensor
    /// is concrete, this registers the tensor as an input
    /// to the graph. If the tensor is virtual, then we merge
    /// the two graphs if they differ.
    fn to_graph(&self, cx: &OpGraphContextHandle) -> NodeId {
        let guard = self.inner.lock();
        match &guard.data {
            Data::Virtual(virt) => {
                if Arc::ptr_eq(&virt.graph, cx) {
                    virt.node
                } else {
                    let cx2 = virt.graph.clone();
                    drop(guard);
                    cx2.lock().merge_into(cx);
                    let guard = self.inner.lock();
                    let Data::Virtual(virt) = &guard.data else {
                        unreachable!()
                    };
                    virt.node
                }
            }
            Data::Concrete(_) => {
                cx.lock()
                    .new_input(guard.shape().len() as u32, guard.data_type(), &self.inner)
            }
        }
    }

    fn op_binary_pointwise(&self, rhs: &Self, op: BinaryPointwiseOp) -> Self {
        let (cx, this) = self.make_graph();
        Tensor::from_op(
            &cx,
            Op::BinaryPointwise(BinaryPointwise {
                lhs: this,
                rhs: rhs.to_graph(&cx),
                op,
            }),
        )
    }

    fn op_unary_pointwise(&self, op: UnaryPointwiseOp) -> Self {
        let (cx, this) = self.make_graph();
        Tensor::from_op(&cx, Op::UnaryPointwise(UnaryPointwise { input: this, op }))
    }
}

impl<const D: usize> Neg for Tensor<D> {
    type Output = Tensor<D>;

    fn neg(self) -> Self::Output {
        self.op_unary_pointwise(UnaryPointwiseOp::Neg)
    }
}

impl<const D: usize> Add for Tensor<D> {
    type Output = Tensor<D>;

    fn add(self, rhs: Self) -> Self::Output {
        self.op_binary_pointwise(&rhs, BinaryPointwiseOp::Add)
    }
}

impl<const D: usize> Sub for Tensor<D> {
    type Output = Tensor<D>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl<const D: usize> Mul for Tensor<D> {
    type Output = Tensor<D>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.op_binary_pointwise(&rhs, BinaryPointwiseOp::Mul)
    }
}

struct TensorInner {
    data: Data,
}

impl TensorInner {
    pub fn shape(&self) -> &[usize] {
        match &self.data {
            Data::Virtual(virt) => &virt.shape,
            Data::Concrete(conc) => match conc {
                #[cfg(feature = "cuda")]
                ConcreteData::Cuda(tensor) => tensor.shape(),
                #[cfg(feature = "cpu")]
                ConcreteData::Cpu => todo!(),
            },
        }
    }

    pub fn data_type(&self) -> DataType {
        match &self.data {
            Data::Virtual(virt) => virt.data_type,
            Data::Concrete(conc) => match conc {
                #[cfg(feature = "cuda")]
                ConcreteData::Cuda(tensor) => tensor.data_type(),
                #[cfg(feature = "cpu")]
                ConcreteData::Cpu => todo!(),
            },
        }
    }
}

enum Data {
    Virtual(VirtualData),
    Concrete(ConcreteData),
}

struct VirtualData {
    graph: OpGraphContextHandle,
    node: NodeId,
    shape: Vec<usize>,
    data_type: DataType,
}

enum ConcreteData {
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::tensor::RawTensor),
    #[cfg(feature = "cpu")]
    Cpu,
}

type OpGraphContextHandle = Arc<Mutex<OpGraphContext>>;

#[derive(Default)]
struct OpGraphContext {
    op_graph: OpGraph,
    inputs: SecondaryMap<NodeId, Arc<Mutex<TensorInner>>>,
    node_to_tensor: SecondaryMap<NodeId, Weak<Mutex<TensorInner>>>,
}

impl OpGraphContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn merge_into(&mut self, other_handle: &OpGraphContextHandle) {
        let mut other = other_handle.lock();
        let node_mapping = self.op_graph.merge_into(&mut other.op_graph);

        for (old_node_id, tensor) in &self.node_to_tensor {
            let new_node_id = node_mapping[old_node_id];
            if let Some(tensor) = tensor.upgrade() {
                let mut guard = tensor.lock();
                if let Data::Virtual(virt) = &mut guard.data {
                    virt.graph = Arc::clone(other_handle);
                    virt.node = new_node_id;
                }
            }
        }

        for (old_input_id, tensor) in &self.inputs {
            let new_input_id = node_mapping[old_input_id];
            other.inputs.insert(new_input_id, Arc::clone(tensor));
        }
    }

    pub fn new_input(
        &mut self,
        dimension: u32,
        data_type: DataType,
        tensor: &Arc<Mutex<TensorInner>>,
    ) -> NodeId {
        let id = self.op_graph.new_input(Descriptor {
            dimension,
            data_type,
        });
        self.inputs.insert(id, Arc::clone(tensor));
        id
    }

    pub fn new_op(&mut self, op: Op, owner: &Arc<Mutex<TensorInner>>) -> NodeId {
        let node = self.op_graph.new_op(op);
        self.node_to_tensor.insert(node, Arc::downgrade(owner));
        node
    }

    pub fn resolve(&mut self, device: &Device) {
        // Mark tensors that are still alive as outputs.
        // This ensures these tensors are not virtualized by the backend,
        // as their data may be used later.
        for (node, tensor) in &self.node_to_tensor {
            if tensor.strong_count() > 0 {
                self.op_graph.new_output(node);
            }
        }

        match device {
            #[cfg(feature = "cuda")]
            Device::Cuda(device_index) => cuda::resolve(self, *device_index),
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use crate::cuda::{context::CudaContext, execution::TensorMap};

    pub fn resolve(graph: &mut OpGraphContext, device_index: u32) {
        let cx = CudaContext::global(device_index).expect("failed to create CUDA context");
        let plan = crate::cuda::planner::compile_plan(cx, &Arc::new(graph.op_graph.clone()))
            .expect("failed to generate plan");

        let mut inputs = TensorMap::new();
        for &input in graph.op_graph.inputs() {
            let tensor = graph.inputs[input].lock();
            let Data::Concrete(data) = &tensor.data else {
                panic!("input tensor cannot be virtual")
            };

            // TODO: check device / handle multi-device
            let ConcreteData::Cuda(data) = &data else {
                panic!("data must reside on CUDA")
            };

            inputs.insert(input, data.clone());
        }

        let mut outputs = crate::cuda::execution::execute_plan(&plan, cx, inputs)
            .expect("failed to execute plan");

        for (node, tensor) in &graph.node_to_tensor {
            if let Some(tensor) = tensor.upgrade() {
                tensor.lock().data =
                    Data::Concrete(ConcreteData::Cuda(outputs.remove(node).unwrap()));
            }
        }
    }
}
