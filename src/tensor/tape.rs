use crate::{raw_tensor::RawTensor, Gradients, ParamId, Tensor};
use std::sync::{atomic::AtomicU64, Arc};

/// Stores activations and operations needed for backpropagation.
#[derive(Clone)]
pub struct Tape {
    #[expect(unused)] // TODO use later for gradient checkpointing
    position: Position,
    backprop: Arc<dyn Backprop>,
}

impl Tape {
    pub fn new_constant(activation: RawTensor) -> Self {
        let position = Position::new();
        Self {
            position,
            backprop: Arc::new(EmptyBackprop { activation }),
        }
    }

    pub fn new_param(id: ParamId, value: RawTensor) -> Self {
        let position = Position::new();
        Self {
            position,
            backprop: Arc::new(ParamBackprop { id, value }),
        }
    }

    pub fn checkpoint(self, value: RawTensor) -> Self {
        let position = Position::new();
        Self {
            position,
            backprop: Arc::new(Checkpoint {
                backprop_prev: self.backprop,
                activation: value,
            }),
        }
    }

    pub fn append_unary(
        self,
        op: impl Fn(RawTensor) -> RawTensor + Send + Sync + 'static,
        compute_flow: impl Fn(RawTensor, RawTensor) -> RawTensor + Send + Sync + 'static,
    ) -> Tape {
        Tape {
            position: Position::new(),
            backprop: Arc::new(UnaryBackprop {
                backprop_prev: self.backprop,
                op: Box::new(op),
                compute_flow: Box::new(compute_flow),
            }),
        }
    }

    pub fn append_binary(
        self,
        rhs: Tape,
        op: impl Fn(RawTensor, RawTensor) -> RawTensor + Send + Sync + 'static,
        compute_flow1: impl Fn(RawTensor, RawTensor, RawTensor) -> RawTensor + Send + Sync + 'static,
        compute_flow2: impl Fn(RawTensor, RawTensor, RawTensor) -> RawTensor + Send + Sync + 'static,
    ) -> Tape {
        Tape {
            position: Position::new(),
            backprop: Arc::new(BinaryBackprop {
                backprop_prev1: self.backprop,
                backprop_prev2: rhs.backprop,
                op: Box::new(op),
                compute_flow1: Box::new(compute_flow1),
                compute_flow2: Box::new(compute_flow2),
            }),
        }
    }

    pub fn backward<const D: usize>(&self, val: Tensor<D>) -> Gradients {
        let mut grads = Gradients::new();
        self.backprop.backprop(&mut grads, val.raw);
        grads
    }
}

trait Backprop: Send + Sync + 'static {
    fn activation(&self) -> RawTensor;
    fn backprop(&self, gradients: &mut Gradients, flow: RawTensor);
}

struct EmptyBackprop {
    activation: RawTensor,
}

impl Backprop for EmptyBackprop {
    fn activation(&self) -> RawTensor {
        self.activation.clone()
    }

    fn backprop(&self, gradients: &mut Gradients, flow: RawTensor) {
        let _ = (gradients, flow);
    }
}

struct ParamBackprop {
    id: ParamId,
    value: RawTensor,
}

impl Backprop for ParamBackprop {
    fn activation(&self) -> RawTensor {
        self.value.clone()
    }

    fn backprop(&self, gradients: &mut Gradients, flow: RawTensor) {
        gradients.insert_raw(self.id, flow);
    }
}

struct UnaryBackprop {
    backprop_prev: Arc<dyn Backprop>,
    op: Box<dyn Fn(RawTensor) -> RawTensor + Send + Sync>,
    compute_flow: Box<dyn Fn(RawTensor, RawTensor) -> RawTensor + Send + Sync>,
}

impl Backprop for UnaryBackprop {
    fn activation(&self) -> RawTensor {
        (self.op)(self.backprop_prev.activation())
    }

    fn backprop(&self, gradients: &mut Gradients, flow: RawTensor) {
        let flow = (self.compute_flow)(self.backprop_prev.activation(), flow);
        self.backprop_prev.backprop(gradients, flow);
    }
}

struct BinaryBackprop {
    backprop_prev1: Arc<dyn Backprop>,
    backprop_prev2: Arc<dyn Backprop>,
    op: Box<dyn Fn(RawTensor, RawTensor) -> RawTensor + Send + Sync>,
    compute_flow1: Box<dyn Fn(RawTensor, RawTensor, RawTensor) -> RawTensor + Send + Sync>,
    compute_flow2: Box<dyn Fn(RawTensor, RawTensor, RawTensor) -> RawTensor + Send + Sync>,
}

impl Backprop for BinaryBackprop {
    fn activation(&self) -> RawTensor {
        (self.op)(
            self.backprop_prev1.activation(),
            self.backprop_prev2.activation(),
        )
    }

    fn backprop(&self, gradients: &mut Gradients, flow: RawTensor) {
        let activation1 = self.backprop_prev1.activation();
        let activation2 = self.backprop_prev2.activation();
        let flow1 = (self.compute_flow1)(activation1.clone(), activation2.clone(), flow.clone());
        let flow2 = (self.compute_flow2)(activation1, activation2, flow);

        self.backprop_prev1.backprop(gradients, flow1);
        self.backprop_prev2.backprop(gradients, flow2);
    }
}

struct Checkpoint {
    backprop_prev: Arc<dyn Backprop>,
    activation: RawTensor,
}

impl Backprop for Checkpoint {
    fn activation(&self) -> RawTensor {
        self.activation.clone()
    }

    fn backprop(&self, gradients: &mut Gradients, flow: RawTensor) {
        self.backprop_prev.backprop(gradients, flow);
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
