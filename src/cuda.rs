use crate::{
    backend::{Backend, Executor, Instruction, Plan, TensorMap},
    cuda::{
        allocator::{Memory, StreamId},
        context::{CudaContext, CudaEvent, CudaStream},
        instr::{cudnn_graph::CudnnGraph, Instr},
        tensor_storage::{TensorStorage, TensorStorageId},
    },
    data_type::{DataType, DataVec},
    opgraph::{op::Op, subgraph::OpSubgraph, NodeId, OpGraph},
};
use ahash::AHashSet;
use cudarc::{
    driver,
    driver::sys::{CUevent, CUevent_flags_enum, CUevent_wait_flags_enum},
};
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
            Op::Matmul(_) => {
                let subgraph = OpSubgraph::from_nodes(graph, vec![node_id]);
                Instr::CudnnGraph(CudnnGraph::new(subgraph))
            }
        }
    }

    fn create_tensor_with_data(&self, data: DataVec, device: Self::Device) -> Self::TensorStorage {
        let cx = CudaContext::global(device).expect("failed to get CUDA context");
        let tensor = TensorStorage::new(data.data_type(), data.len(), cx, None)
            .expect("failed to allocate tensor");
        tensor
            .initialize_with_data(&data, cx.htod_stream())
            .expect("failed to initialize tensor with data");
        tensor
            .ready_event()
            .record(cx.htod_stream())
            .expect("failed to record event");

        let event = tensor.ready_event().clone();

        // `data` needs to live until the transfer completes.
        // for now, we implement this by dropping the vec
        // on a thread once the event completes.
        rayon::spawn(move || {
            event.sync().expect("failed to sync on event");
            drop(data);
        });

        tensor
    }

    fn download_tensor(
        &self,
        tensor: &Self::TensorStorage,
        callback: impl FnOnce(DataVec) + Send + Sync + 'static,
        device: Self::Device,
    ) {
        let cx = CudaContext::global(device).expect("failed to get CUDA context");
        let event = CudaEvent::new().expect("failed to create CUDA event");

        tensor
            .ready_event()
            .wait(cx.dtoh_stream())
            .expect("failed to wait event");
        let data = unsafe {
            tensor
                .async_copy_to_host(cx.dtoh_stream())
                .expect("failed to copy tensor to host")
        };
        event
            .record(cx.dtoh_stream())
            .expect("failed to record event");

        // Keep tensor alive until download completes
        let tensor_clone = tensor.clone();

        rayon::spawn(move || {
            event.sync().expect("failed to sync event");
            callback(data);
            drop(tensor_clone);
        });
    }

    fn begin_execute(
        &self,
        input_tensors: &TensorMap<Self>,
        device: Self::Device,
        plan: &Plan<Self>,
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
        let sync_events = streams
            .iter()
            .map(|_| {
                driver::result::event::create(CUevent_flags_enum::CU_EVENT_DEFAULT)
                    .expect("failed to create event")
            })
            .collect();

        let start = CudaEvent::new().expect("failed to create CUDA event");
        let stop = CudaEvent::new().expect("failed to create CUDA event");

        start
            .record(&streams[0])
            .expect("failed to record CUDA event");

        CudaExecutor {
            cx,
            streams,
            sync_events,
            instr_index: 0,
            hold_allocations: Vec::new(),
            allocation_stream,
            synced_tensors: AHashSet::new(),
            synced_tensors_this_step: Vec::new(),
            start,
            stop,
        }
    }
}

pub struct CudaExecutor {
    cx: &'static CudaContext,
    streams: &'static [CudaStream],
    sync_events: Vec<CUevent>,
    hold_allocations: Vec<Memory>,
    instr_index: usize,
    allocation_stream: StreamId,
    synced_tensors: AHashSet<TensorStorageId>,
    synced_tensors_this_step: Vec<TensorStorageId>,

    /// Events used for profiling.
    #[cfg_attr(not(feature = "cuda-tracing"), expect(unused))]
    start: CudaEvent,
    stop: CudaEvent,
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
        let stream = &self.streams[self.instr_index % self.streams.len()];

        for input in instr.inputs() {
            let storage = tensors.get_storage(input);
            if !self.synced_tensors.contains(&storage.id()) {
                storage
                    .ready_event()
                    .wait(stream)
                    .expect("failed to wait on event");
                self.synced_tensors_this_step.push(storage.id());
            }
        }

        instr.execute(tensors, stream, self.cx, &mut self.hold_allocations);

        self.instr_index += 1;

        for output in instr.outputs() {
            self.synced_tensors.insert(tensors.get_storage(output).id());
            tensors
                .get_storage(output)
                .ready_event()
                .record(stream)
                .expect("failed to record event");
        }
    }

    #[profiling::function]
    fn end_step(&mut self) {
        for (event, stream) in self.sync_events.iter().zip(self.streams) {
            unsafe {
                driver::result::event::record(*event, stream.raw())
                    .expect("failed to record event");
            }
        }
        for (i, stream) in self.streams.iter().enumerate() {
            for (j, event) in self.sync_events.iter().enumerate() {
                if i != j {
                    unsafe {
                        driver::result::stream::wait_event(
                            stream.raw(),
                            *event,
                            CUevent_wait_flags_enum::CU_EVENT_WAIT_DEFAULT,
                        )
                        .expect("failed to wait on event");
                    }
                }
            }
        }
        self.synced_tensors
            .extend(self.synced_tensors_this_step.drain(..));
    }
}

impl Drop for CudaExecutor {
    fn drop(&mut self) {
        self.stop
            .record(&self.streams[0])
            .expect("failed to record stop event");
        #[cfg(feature = "cuda-tracing")]
        {
            let time_elapsed = self.stop.measure_time_elapsed(&self.start).unwrap();
            tracing::debug!("GPU execution time: {time_elapsed:.2?}");
        }

        for event in self.sync_events.drain(..) {
            unsafe {
                driver::result::event::destroy(event).expect("failed to destroy event");
            }
        }
        self.hold_allocations.clear();
        self.cx
            .set_previous_alloc_stream_for_thread(self.allocation_stream);
        self.cx.allocator().end_stream(self.allocation_stream);
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
