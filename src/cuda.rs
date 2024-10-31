use crate::{
    backend::{Backend, Executor, Instruction, Plan, TensorMap},
    cuda::{
        allocator::{DeviceMemory, StreamId},
        context::{CudaContext, CudaEvent, CudaStream},
        instr::{cudnn_graph::CudnnGraph, permute_dims::PermuteDims, Instr},
        tensor_storage::{TensorStorage, TensorStorageId},
    },
    data_type::{DataSlice, DataType, DataVec},
    opgraph::{op::Op, subgraph::OpSubgraph, NodeId, OpGraph},
};
use ahash::AHashSet;
use instr::pointwise::PointwiseGraph;
use std::sync::Arc;

mod allocator;
pub mod context;
mod cudnn;
pub mod error;
mod instr;
mod kernel_jit;
mod tensor_storage;

#[derive(Copy, Clone, Debug)]
pub struct Cuda;

impl Backend for Cuda {
    type Device = u32;
    type Instr = Instr;
    type TensorStorage = TensorStorage;
    type Executor = CudaExecutor;

    fn make_instr_for_op(&self, op: &Op, graph: &Arc<OpGraph>, node_id: NodeId) -> Self::Instr {
        match op {
            // Heuristic: the PermuteDims JIT kernels offer fewer fusion
            // opportunities, but perform much better than pointwise JIT
            // for large shapes.
            Op::SwapDims(_) if graph.get(node_id).descriptor().shape.num_elements() > 1_000_000 => {
                let subgraph = OpSubgraph::from_nodes(graph, vec![node_id]);
                Instr::PermuteDims(PermuteDims::new(subgraph))
            }
            Op::SwapDims(_)
            | Op::UnaryPointwise(_)
            | Op::BinaryPointwise(_)
            | Op::ChangeDataType(_)
            | Op::Reduce(_)
            | Op::Broadcast(_)
            | Op::Reshape(_)
            | Op::Compare(_)
            | Op::Select(_) => {
                let subgraph = OpSubgraph::from_nodes(graph, vec![node_id]);
                Instr::PointwiseGraph(PointwiseGraph::new(subgraph))
            }
            Op::Matmul(_) | Op::Conv(_) | Op::ConvBackwardData(_) | Op::ConvBackwardFilter(_) => {
                let subgraph = OpSubgraph::from_nodes(graph, vec![node_id]);
                Instr::CudnnGraph(CudnnGraph::new(subgraph))
            }
        }
    }

    fn create_tensor_with_data(
        &self,
        data: DataSlice,
        device: Self::Device,
    ) -> Self::TensorStorage {
        let cx = CudaContext::global(device).expect("failed to get CUDA context");
        let tensor = TensorStorage::new(data.data_type(), data.len(), cx, None)
            .expect("failed to allocate tensor");
        tensor
            .initialize_with_data(data, cx.htod_stream())
            .expect("failed to initialize tensor with data");
        tensor
            .ready_event()
            .record(cx.htod_stream())
            .expect("failed to record event");

        tensor
    }

    fn download_tensor(
        &self,
        tensor: &Self::TensorStorage,
        callback: impl FnOnce(DataVec) + Send + 'static,
        device: Self::Device,
    ) {
        let cx = CudaContext::global(device).expect("failed to get CUDA context");
        tensor
            .ready_event()
            .wait(cx.dtoh_stream())
            .expect("failed to wait event");

        tensor
            .async_copy_to_host(cx.dtoh_stream(), callback)
            .expect("failed to perform tensor device=>host copy");
    }

    fn begin_execute(
        &self,
        input_tensors: &TensorMap<Self>,
        device: Self::Device,
        plan: &Plan<Self>,
        _op_graph: &Arc<OpGraph>,
    ) -> Self::Executor {
        let cx = CudaContext::global(device).expect("failed to get CUDA context");

        parallel_compile_kernels(plan, cx);

        let allocation_stream = cx
            .allocator()
            .begin_stream(cx.previous_alloc_stream_for_thread());

        for (_, tensor) in input_tensors.storages() {
            tensor.mark_in_use_by_stream(allocation_stream);
        }

        let streams = cx.stream_pool().expect("failed to get stream pool");
        let op_stream = &streams[0];

        let start = CudaEvent::new().expect("failed to create CUDA event");
        let stop = CudaEvent::new().expect("failed to create CUDA event");

        start
            .record(op_stream)
            .expect("failed to record CUDA event");

        CudaExecutor {
            cx,
            op_stream,
            instr_index: 0,
            hold_allocations: Vec::new(),
            allocation_stream,
            synced_tensors: AHashSet::new(),
            synced_tensors_this_step: Vec::new(),
            start: Arc::new(start),
            stop: Arc::new(stop),
            output_tensors: enumerate_output_tensors(plan),
        }
    }
}

fn enumerate_output_tensors(plan: &Plan<Cuda>) -> AHashSet<NodeId> {
    let mut alive = AHashSet::new();
    for step in plan.steps() {
        for instr in step.instrs() {
            alive.extend(instr.outputs());
        }
        for free in step.tensors_to_release() {
            alive.remove(&free);
        }
    }
    alive
}

pub struct CudaExecutor {
    cx: &'static CudaContext,
    op_stream: &'static CudaStream,
    hold_allocations: Vec<DeviceMemory>,
    instr_index: usize,
    allocation_stream: StreamId,
    synced_tensors: AHashSet<TensorStorageId>,
    synced_tensors_this_step: Vec<TensorStorageId>,

    output_tensors: AHashSet<NodeId>,

    /// Events used for profiling.
    #[cfg_attr(not(feature = "cuda-tracing"), expect(unused))]
    start: Arc<CudaEvent>,
    stop: Arc<CudaEvent>,
}

impl Executor<Cuda> for CudaExecutor {
    fn begin_step(&mut self) {
        self.instr_index = 0;
        self.hold_allocations.clear();
    }

    #[profiling::function]
    fn allocate_tensor(
        &self,
        device: <Cuda as Backend>::Device,
        data_type: DataType,
        len: usize,
    ) -> <Cuda as Backend>::TensorStorage {
        let cx = CudaContext::global(device).unwrap();
        TensorStorage::new(data_type, len, cx, Some(self.allocation_stream))
            .unwrap_or_else(|e| panic!("failed to allocate tensor for {len}x {data_type:?}: {e}"))
    }

    #[profiling::function]
    fn execute_instr(&mut self, instr: &Instr, tensors: &mut TensorMap<Cuda>) {
        for input in instr.inputs() {
            let storage = tensors.get_storage(input);
            if !self.synced_tensors.contains(&storage.id()) {
                storage
                    .ready_event()
                    .wait(self.op_stream)
                    .expect("failed to wait on event");
                self.synced_tensors_this_step.push(storage.id());
            }
        }

        instr.execute(
            tensors,
            self.op_stream,
            self.cx,
            &mut self.hold_allocations,
            self.allocation_stream,
        );

        self.instr_index += 1;

        for output in instr.outputs() {
            self.synced_tensors.insert(tensors.get_storage(output).id());
            if self.output_tensors.contains(&output) {
                tensors
                    .get_storage(output)
                    .ready_event()
                    .record(self.op_stream)
                    .expect("failed to record event");
            }
        }
    }

    #[profiling::function]
    fn end_step(&mut self) {
        self.synced_tensors
            .extend(self.synced_tensors_this_step.drain(..));
    }
}

impl Drop for CudaExecutor {
    fn drop(&mut self) {
        self.stop
            .record(self.op_stream)
            .expect("failed to record stop event");
        #[cfg(feature = "cuda-tracing")]
        {
            let stop = self.stop.clone();
            let start = self.start.clone();
            crate::thread_pool::BlockingThreadPool::global().spawn(move || {
                let time_elapsed = stop.measure_time_elapsed(&start).unwrap();
                tracing::debug!("GPU execution time: {time_elapsed:.2?}");
            });
        }

        self.hold_allocations.clear();
        self.cx
            .set_previous_alloc_stream_for_thread(self.allocation_stream);

        let cx = self.cx;
        let allocation_stream = self.allocation_stream;

        self.op_stream
            .insert_host_callback(move || {
                cx.allocator().end_stream(allocation_stream);
            })
            .unwrap();
    }
}

fn parallel_compile_kernels(plan: &Plan<Cuda>, cx: &CudaContext) {
    use rayon::prelude::*;
    plan.steps()
        .par_iter()
        .flat_map(|step| step.instrs().par_iter())
        .for_each(|instr| {
            instr.precompile(cx);
        });
}
