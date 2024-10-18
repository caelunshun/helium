use crate::{
    backend::{Backend, Executor, TensorMap},
    cuda::{
        allocator::{Memory, StreamId},
        context::{CudaContext, CudaStream},
        instr::{cudnn_graph::CudnnGraph, Instr},
        tensor_storage::TensorStorage,
    },
    data_type::{DataType, DataTypeConversion},
    opgraph::{op::Op, subgraph::OpSubgraph, NodeId, OpGraph},
};
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
            Op::UploadTensor(op) => Instr::UploadTensor {
                node: node_id,
                data: op.data.clone(),
            },
            Op::SwapDims(_)
            | Op::UnaryPointwise(_)
            | Op::BinaryPointwise(_)
            | Op::ChangeDataType(_)
            | Op::Reduce(_)
            | Op::Broadcast(_)
            | Op::Reshape(_) => {
                let subgraph = OpSubgraph::from_nodes(graph, vec![node_id]);
                Instr::PointwiseGraph(PointwiseGraph::new(subgraph))
            }
            Op::Matmul(_) => {
                let subgraph = OpSubgraph::from_nodes(graph, vec![node_id]);
                Instr::CudnnGraph(CudnnGraph::new(subgraph))
            }
        }
    }

    fn begin_execute(&self, device: Self::Device) -> Self::Executor {
        let cx = CudaContext::global(device).expect("failed to get CUDA context");
        let streams = cx.stream_pool().expect("failed to get stream pool");
        let sync_events = streams
            .iter()
            .map(|_| {
                driver::result::event::create(CUevent_flags_enum::CU_EVENT_DEFAULT)
                    .expect("failed to create event")
            })
            .collect();
        CudaExecutor {
            cx,
            streams,
            sync_events,
            instr_index: 0,
            hold_allocations: Vec::new(),
            allocation_stream: cx.allocator().begin_stream(),
        }
    }

    fn tensor_to_vec<E: DataTypeConversion>(&self, tensor: &Self::TensorStorage) -> Vec<E> {
        tensor.to_vec()
    }
}

pub struct CudaExecutor {
    cx: &'static CudaContext,
    streams: &'static [CudaStream],
    sync_events: Vec<CUevent>,
    hold_allocations: Vec<Memory>,
    instr_index: usize,
    allocation_stream: StreamId,
}

impl Executor<Cuda> for CudaExecutor {
    fn begin_step(&mut self) {
        self.instr_index = 0;
        self.hold_allocations.clear();
    }

    fn execute_instr(&mut self, instr: &Instr, tensors: &mut TensorMap<Cuda>) {
        let stream = &self.streams[self.instr_index % self.streams.len()];

        instr.execute(tensors, stream, self.cx, &mut self.hold_allocations);

        self.instr_index += 1;
    }

    fn allocate_tensor(
        &self,
        device: <Cuda as Backend>::Device,
        data_type: DataType,
        len: usize,
    ) -> <Cuda as Backend>::TensorStorage {
        let cx = CudaContext::global(device).unwrap();
        TensorStorage::new(data_type, len, cx, self.allocation_stream)
            .unwrap_or_else(|e| panic!("failed to allocate tensor for {len}x {data_type:?}: {e}"))
    }

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
    }
}

impl Drop for CudaExecutor {
    fn drop(&mut self) {
        for stream in self.streams {
            unsafe {
                driver::result::stream::synchronize(stream.raw())
                    .expect("failed to synchronize stream");
            }
        }
        for event in self.sync_events.drain(..) {
            unsafe {
                driver::result::event::destroy(event).expect("failed to destroy event");
            }
        }
        self.hold_allocations.clear();
        self.cx.allocator().end_stream(self.allocation_stream);
    }
}
