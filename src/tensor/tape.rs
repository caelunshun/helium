use crate::{raw_tensor::RawTensor, Gradients, ParamId, Tensor};
use ahash::{AHashMap, AHashSet};
use std::{
    collections::hash_map::Entry,
    sync::{atomic::AtomicU64, Arc},
};

/// Stores activations and operations needed for backpropagation.
#[derive(Clone)]
pub struct Tape {
    backprop: Arc<dyn Backprop>,
}

impl Tape {
    pub fn new_constant(activation: RawTensor) -> Self {
        Self {
            backprop: Arc::new(ConstantBackprop {
                position: Position::new(),
                activation,
            }),
        }
    }

    pub fn new_param(id: ParamId, value: RawTensor) -> Self {
        Self {
            backprop: Arc::new(ParamBackprop {
                position: Position::new(),
                id,
                value,
            }),
        }
    }

    pub fn checkpoint(self, value: RawTensor) -> Self {
        Self {
            backprop: Arc::new(Checkpoint {
                position: Position::new(),
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
            backprop: Arc::new(UnaryBackprop {
                position: Position::new(),
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
            backprop: Arc::new(BinaryBackprop {
                position: Position::new(),
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
        let mut state = State::new(&self.backprop);
        self.backprop.backprop(&mut state, &mut grads, val.raw);
        grads
    }
}

struct State {
    memoized_activations: AHashMap<Position, RawTensor>,
    flow_accumulation: AHashMap<Position, RawTensor>,
    remaining_dependency_counts: AHashMap<Position, usize>,
}

impl State {
    pub fn new(root: &Arc<dyn Backprop>) -> Self {
        let mut dependency_counts = AHashMap::new();
        let mut stack = vec![root.clone()];
        let mut visited = AHashSet::new();
        while let Some(node) = stack.pop() {
            if !visited.insert(node.position()) {
                continue;
            }

            for child in node.children() {
                *dependency_counts.entry(child.position()).or_insert(0) += 1;
                stack.push(child);
            }
        }

        Self {
            remaining_dependency_counts: dependency_counts,
            flow_accumulation: AHashMap::new(),
            memoized_activations: AHashMap::new(),
        }
    }

    pub fn get_activation(&mut self, child: &dyn Backprop) -> RawTensor {
        if let Some(x) = self.memoized_activations.get(&child.position()) {
            x.clone()
        } else {
            let x = child.activation(self);
            self.memoized_activations
                .insert(child.position(), x.clone());
            x
        }
    }

    pub fn propagate_flow(
        &mut self,
        child: &dyn Backprop,
        gradients: &mut Gradients,
        flow: RawTensor,
    ) {
        let remaining_count = self
            .remaining_dependency_counts
            .get_mut(&child.position())
            .unwrap();
        *remaining_count -= 1;

        match self.flow_accumulation.entry(child.position()) {
            Entry::Vacant(entry) => {
                entry.insert(flow);
            }
            Entry::Occupied(mut entry) => {
                let old = entry.get().clone();
                entry.insert(old + flow);
            }
        }

        if *remaining_count == 0 {
            let flow = self.flow_accumulation.remove(&child.position()).unwrap();
            child.backprop(self, gradients, flow);
        }
    }
}

trait Backprop: Send + Sync + 'static {
    fn position(&self) -> Position;
    fn children(&self) -> Vec<Arc<dyn Backprop>>;
    fn activation(&self, state: &mut State) -> RawTensor;
    fn backprop(&self, state: &mut State, gradients: &mut Gradients, flow: RawTensor);
}

struct ConstantBackprop {
    position: Position,
    activation: RawTensor,
}

impl Backprop for ConstantBackprop {
    fn position(&self) -> Position {
        self.position
    }

    fn children(&self) -> Vec<Arc<dyn Backprop>> {
        vec![]
    }

    fn activation(&self, _state: &mut State) -> RawTensor {
        self.activation.clone()
    }

    fn backprop(&self, state: &mut State, gradients: &mut Gradients, flow: RawTensor) {
        let _ = (state, gradients, flow);
    }
}

struct ParamBackprop {
    position: Position,
    id: ParamId,
    value: RawTensor,
}

impl Backprop for ParamBackprop {
    fn position(&self) -> Position {
        self.position
    }

    fn children(&self) -> Vec<Arc<dyn Backprop>> {
        vec![]
    }

    fn activation(&self, _state: &mut State) -> RawTensor {
        self.value.clone()
    }

    fn backprop(&self, _state: &mut State, gradients: &mut Gradients, flow: RawTensor) {
        gradients.insert_raw(self.id, flow);
    }
}

struct UnaryBackprop {
    position: Position,
    backprop_prev: Arc<dyn Backprop>,
    op: Box<dyn Fn(RawTensor) -> RawTensor + Send + Sync>,
    compute_flow: Box<dyn Fn(RawTensor, RawTensor) -> RawTensor + Send + Sync>,
}

impl Backprop for UnaryBackprop {
    fn position(&self) -> Position {
        self.position
    }

    fn children(&self) -> Vec<Arc<dyn Backprop>> {
        vec![self.backprop_prev.clone()]
    }

    fn activation(&self, state: &mut State) -> RawTensor {
        let prev = state.get_activation(&*self.backprop_prev);
        (self.op)(prev)
    }

    fn backprop(&self, state: &mut State, gradients: &mut Gradients, flow: RawTensor) {
        let input = state.get_activation(&*self.backprop_prev);
        let flow = (self.compute_flow)(input, flow);
        state.propagate_flow(&*self.backprop_prev, gradients, flow);
    }
}

struct BinaryBackprop {
    position: Position,
    backprop_prev1: Arc<dyn Backprop>,
    backprop_prev2: Arc<dyn Backprop>,
    op: Box<dyn Fn(RawTensor, RawTensor) -> RawTensor + Send + Sync>,
    compute_flow1: Box<dyn Fn(RawTensor, RawTensor, RawTensor) -> RawTensor + Send + Sync>,
    compute_flow2: Box<dyn Fn(RawTensor, RawTensor, RawTensor) -> RawTensor + Send + Sync>,
}

impl Backprop for BinaryBackprop {
    fn position(&self) -> Position {
        self.position
    }

    fn children(&self) -> Vec<Arc<dyn Backprop>> {
        vec![self.backprop_prev1.clone(), self.backprop_prev2.clone()]
    }

    fn activation(&self, state: &mut State) -> RawTensor {
        let a = state.get_activation(&*self.backprop_prev1);
        let b = state.get_activation(&*self.backprop_prev2);
        (self.op)(a, b)
    }

    fn backprop(&self, state: &mut State, gradients: &mut Gradients, flow: RawTensor) {
        let activation1 = state.get_activation(&*self.backprop_prev1);
        let activation2 = state.get_activation(&*self.backprop_prev2);
        let flow1 = (self.compute_flow1)(activation1.clone(), activation2.clone(), flow.clone());
        let flow2 = (self.compute_flow2)(activation1, activation2, flow);

        state.propagate_flow(&*self.backprop_prev1, gradients, flow1);
        state.propagate_flow(&*self.backprop_prev2, gradients, flow2);
    }
}

struct Checkpoint {
    position: Position,
    backprop_prev: Arc<dyn Backprop>,
    activation: RawTensor,
}

impl Backprop for Checkpoint {
    fn position(&self) -> Position {
        self.position
    }

    fn children(&self) -> Vec<Arc<dyn Backprop>> {
        vec![self.backprop_prev.clone()]
    }

    fn activation(&self, _state: &mut State) -> RawTensor {
        self.activation.clone()
    }

    fn backprop(&self, state: &mut State, gradients: &mut Gradients, flow: RawTensor) {
        state.propagate_flow(&*self.backprop_prev, gradients, flow);
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
