use crate::cuda::{
    allocator::block::{Block, BlockAllocator, PageId},
    error::CudaError,
};
use cudarc::{
    driver,
    driver::sys::{CUcontext, CUdeviceptr},
};
use parking_lot::Mutex;
use slotmap::SecondaryMap;
use std::sync::Arc;

mod block;

pub use block::StreamId;

/// Pooling allocator for a CUDA device.
///
/// We use an aggressive allocator strategy that currently
/// never returns memory to the driver.
pub struct CudaAllocator {
    context: CUcontext,
    block_allocator: BlockAllocator,
    pages: SecondaryMap<PageId, Page>,
    dropped_memories: Arc<Mutex<Vec<Block>>>,
}

unsafe impl Send for CudaAllocator {}
unsafe impl Sync for CudaAllocator {}

impl CudaAllocator {
    /// Maximum alignment that can be requested.
    pub const MAX_ALIGN: u64 = 256; // alignment of cudaMalloc
    /// Minimum size of a page. Pages can be larger to
    /// accommodate larger single allocations.
    const MIN_PAGE_SIZE: u64 = 1024 * 1024 * 1024; // 1 GiB

    /// # Safety
    /// `context` must outlive `self`.
    pub unsafe fn new(context: CUcontext) -> Self {
        Self {
            pages: SecondaryMap::default(),
            block_allocator: BlockAllocator::new(),
            dropped_memories: Arc::new(Mutex::new(Vec::new())),
            context,
        }
    }

    /// Allocate some memory.
    ///
    /// Note: when dropped on the host side,
    /// the freed memory becomes immediately available
    /// to all threads. When used in combination
    /// with asynchronous CUDA streams, this can cause bugs as the GPU
    /// may still be using the allocated memory. In this case,
    /// use `allocate_in_stream` instead.
    pub fn allocate(&mut self, size: u64, align: u64) -> Result<Memory, CudaError> {
        self.allocate_internal(size, align, None)
    }

    /// Allocate some memory with lifetime at least
    /// as long as the given stream.
    ///
    /// When dropped, the freed memory only becomes available
    /// to allocations in the same stream. Other streams
    /// cannot use the memory until `end_stream()` is called,
    /// indicating that the GPU has finished use of the memory.
    pub fn allocate_in_stream(
        &mut self,
        size: u64,
        align: u64,
        stream: StreamId,
    ) -> Result<Memory, CudaError> {
        self.allocate_internal(size, align, Some(stream))
    }

    /// Creates a new stream ID for use with `allocate_in_stream`.
    ///
    /// If `parent` is `Some`, this signals that the parent stream
    /// will complete before this new stream begins execution.
    pub fn begin_stream(&mut self, parent: Option<StreamId>) -> StreamId {
        self.block_allocator.begin_stream(parent)
    }

    /// Signals that a stream has completed execution and thus
    /// any memory allocated on that stream that has been freed
    /// can be reused by other streams.
    pub fn end_stream(&mut self, stream: StreamId) {
        self.block_allocator.end_stream(stream);
    }

    fn allocate_internal(
        &mut self,
        size: u64,
        align: u64,
        stream: Option<StreamId>,
    ) -> Result<Memory, CudaError> {
        assert!(size > 0);
        assert!(align > 0);
        assert!(align.is_power_of_two());
        assert!(align <= Self::MAX_ALIGN);

        unsafe {
            driver::result::ctx::set_current(self.context)?;
        }

        self.process_dropped_memories();

        let block: Block = match self.block_allocator.allocate(size, align, stream) {
            Some(b) => b,
            None => {
                self.allocate_page_for_at_least(size)?;
                self.block_allocator
                    .allocate(size, align, stream)
                    .expect("new page should accommodate allocation")
            }
        };
        let page = &self.pages[block.page];
        let ptr = page.ptr + block.start;

        debug_assert!(block.start + size < page.size);

        Ok(Memory {
            ptr,
            len: size,
            block: Mutex::new(block),
            on_drop: Arc::clone(&self.dropped_memories),
        })
    }

    fn process_dropped_memories(&mut self) {
        for block in self.dropped_memories.lock().drain(..) {
            self.block_allocator.deallocate(block);
        }
    }

    fn allocate_page_for_at_least(&mut self, min_size: u64) -> Result<&Page, CudaError> {
        let size = min_size.max(Self::MIN_PAGE_SIZE).next_power_of_two();

        let ptr = unsafe { driver::result::malloc_sync(size.try_into().unwrap())? };
        unsafe {
            driver::result::memset_d8_sync(ptr, 0, size.try_into().unwrap())?;
        }

        let page = Page { size, ptr };
        let page_id = self.block_allocator.add_page(size);
        self.pages.insert(page_id, page);
        Ok(&self.pages[page_id])
    }
}

/// An allocated region of memory. Freed
/// automatically on drop.
pub struct Memory {
    ptr: CUdeviceptr,
    len: u64,
    /// Block returned from the block allocator
    block: Mutex<Block>,
    on_drop: Arc<Mutex<Vec<Block>>>,
}

impl Memory {
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn mark_in_use_by_stream(&self, stream: StreamId) {
        self.block.lock().mark_in_use_by_stream(stream);
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        self.on_drop.lock().push(self.block.get_mut().clone());
    }
}

/// Allocated page of CUDA memory.
struct Page {
    ptr: CUdeviceptr,
    size: u64,
}

impl Drop for Page {
    fn drop(&mut self) {
        unsafe {
            driver::result::free_sync(self.ptr).expect("failed to free page");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    #[test]
    fn stress_test() {
        let device = CudaDevice::new(0).unwrap();
        let mut allocator = unsafe { CudaAllocator::new(*device.cu_primary_ctx()) };

        let mut rng = Pcg64Mcg::seed_from_u64(66);

        let mut allocated_memories: Vec<Memory> = Vec::new();

        for _ in 0..10_000 {
            if rng.gen_bool(0.5) || allocated_memories.is_empty() {
                let size = rng.gen_range(4..=64 * 1024 * 1024);
                let align = rng.gen_range(1u64..=256).next_power_of_two();
                let memory = allocator.allocate(size, align).unwrap();
                assert!(!allocated_memories.iter().any(|mem| {
                    (mem.device_ptr() <= memory.device_ptr()
                        && memory.device_ptr() < mem.device_ptr() + mem.len())
                        || (memory.device_ptr() <= mem.device_ptr()
                            && mem.device_ptr() < memory.device_ptr() + memory.len())
                }));
                allocated_memories.push(memory);
            } else {
                let i = rng.gen_range(0..allocated_memories.len());
                allocated_memories.swap_remove(i);
            }
        }
    }
}
