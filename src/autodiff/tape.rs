use crate::{
    autodiff::{gradients::Gradients, param::ParamId},
    dyn_tensor::DynTensor,
    Tensor,
};
use std::sync::{atomic::AtomicU64, Arc};

/// Stores activations and operations needed for backpropagation.
#[derive(Clone)]
pub struct Tape {
    #[expect(unused)] // TODO use later for gradient checkpointing
    position: Position,
    backprop: Arc<dyn Backprop>,
}

impl Tape {
    pub fn new_constant() -> Self {
        let position = Position::new();
        Self {
            position,
            backprop: Arc::new(EmptyBackprop),
        }
    }

    pub fn new_param(id: ParamId) -> Self {
        let position = Position::new();
        Self {
            position,
            backprop: Arc::new(ParamBackprop { id }),
        }
    }

    pub fn append_unary<const D1: usize, const D2: usize>(
        self,
        compute_flow: impl Fn(Tensor<D2>) -> Tensor<D1> + Send + Sync + 'static,
    ) -> Tape {
        Tape {
            position: Position::new(),
            backprop: Arc::new(UnaryBackprop {
                backprop_prev: self.backprop,
                compute_flow: Box::new(move |x| DynTensor::new(compute_flow(x.into_inner()))),
            }),
        }
    }

    pub fn append_binary<const D1: usize, const D2: usize, const D3: usize>(
        self,
        rhs: Tape,
        compute_flow1: impl Fn(Tensor<D3>) -> Tensor<D1> + Send + Sync + 'static,
        compute_flow2: impl Fn(Tensor<D3>) -> Tensor<D2> + Send + Sync + 'static,
    ) -> Tape {
        Tape {
            position: Position::new(),
            backprop: Arc::new(BinaryBackprop {
                backprop_prev1: self.backprop,
                backprop_prev2: rhs.backprop,
                compute_flow1: Box::new(move |x| DynTensor::new(compute_flow1(x.into_inner()))),
                compute_flow2: Box::new(move |x| DynTensor::new(compute_flow2(x.into_inner()))),
            }),
        }
    }

    pub fn backward<const D: usize>(self, val: Tensor<D>) -> Gradients {
        let mut grads = Gradients::new();
        self.backprop.backprop(&mut grads, DynTensor::new(val));
        grads
    }
}

trait Backprop: Send + Sync + 'static {
    fn backprop(&self, gradients: &mut Gradients, flow: DynTensor);
}

struct EmptyBackprop;

impl Backprop for EmptyBackprop {
    fn backprop(&self, gradients: &mut Gradients, flow: DynTensor) {
        let _ = (gradients, flow);
    }
}

struct ParamBackprop {
    id: ParamId,
}

impl Backprop for ParamBackprop {
    fn backprop(&self, gradients: &mut Gradients, flow: DynTensor) {
        gradients.insert_dyn(self.id, flow);
    }
}

struct UnaryBackprop {
    backprop_prev: Arc<dyn Backprop>,
    compute_flow: Box<dyn Fn(DynTensor) -> DynTensor + Send + Sync>,
}

impl Backprop for UnaryBackprop {
    fn backprop(&self, gradients: &mut Gradients, flow: DynTensor) {
        let flow = (self.compute_flow)(flow);
        self.backprop_prev.backprop(gradients, flow);
    }
}

struct BinaryBackprop {
    backprop_prev1: Arc<dyn Backprop>,
    backprop_prev2: Arc<dyn Backprop>,
    compute_flow1: Box<dyn Fn(DynTensor) -> DynTensor + Send + Sync>,
    compute_flow2: Box<dyn Fn(DynTensor) -> DynTensor + Send + Sync>,
}

impl Backprop for BinaryBackprop {
    fn backprop(&self, gradients: &mut Gradients, flow: DynTensor) {
        let flow1 = (self.compute_flow1)(flow.clone());
        let flow2 = (self.compute_flow2)(flow);

        self.backprop_prev1.backprop(gradients, flow1);
        self.backprop_prev2.backprop(gradients, flow2);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Position(u64);

impl Position {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Position(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}
