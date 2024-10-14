use crate::cuda::{
    allocator::block::{Block, BlockAllocator},
    error::CudaError,
};
use cudarc::{
    driver,
    driver::sys::{CUcontext, CUdeviceptr},
};
use parking_lot::Mutex;
use std::{collections::BTreeMap, sync::Arc};

mod block;

/// Pooling allocator for a CUDA device.
///
/// We use an aggressive allocator strategy that currently
/// never returns memory to the driver.
pub struct CudaAllocator {
    context: CUcontext,
    block_allocator: BlockAllocator,
    /// Pages sorted by their virtual starting address.
    pages: BTreeMap<u64, Page>,

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
        let block_allocator = BlockAllocator::new(Self::MIN_PAGE_SIZE);
        Self {
            pages: BTreeMap::new(),
            block_allocator,
            dropped_memories: Arc::new(Mutex::new(Vec::new())),
            context,
        }
    }

    /// Allocate some memory.
    pub fn alloc(&mut self, size: u64, align: u64) -> Result<Memory, CudaError> {
        assert!(size > 0);
        assert!(align > 0);
        assert!(align.is_power_of_two());
        assert!(align <= Self::MAX_ALIGN);

        unsafe {
            driver::result::ctx::set_current(self.context)?;
        }

        self.process_dropped_memories();

        let block = self.block_allocator.alloc(size, align);
        let page = self.pages.range(..=block.addr).last();

        let page = match page {
            Some((_, page)) if block.addr + block.size <= page.start + page.size => page,
            _ => {
                // Need to allocate a new page.
                self.allocate_page_for_at_least(size)?
            }
        };

        let offset_in_page = block.addr - page.start;
        let ptr = page.ptr + offset_in_page;

        debug_assert!(offset_in_page < page.size);
        debug_assert!(offset_in_page + size < page.size,);

        Ok(Memory {
            ptr,
            len: size,
            block,
            on_drop: Arc::clone(&self.dropped_memories),
        })
    }

    fn process_dropped_memories(&mut self) {
        for block in self.dropped_memories.lock().drain(..) {
            self.block_allocator.dealloc(block);
        }
    }

    fn allocate_page_for_at_least(&mut self, min_size: u64) -> Result<&Page, CudaError> {
        let size = min_size.max(Self::MIN_PAGE_SIZE).next_power_of_two();

        let ptr = unsafe { driver::result::malloc_sync(size.try_into().unwrap())? };
        unsafe {
            driver::result::memset_d8_sync(ptr, 0, size.try_into().unwrap())?;
        }

        let start = self
            .pages
            .last_key_value()
            .map(|(_, page)| page.start + page.size)
            .unwrap_or(0);
        self.pages.insert(start, Page { start, size, ptr });
        Ok(&self.pages[&start])
    }
}

/// An allocated region of memory. Freed
/// automatically on drop.
pub struct Memory {
    ptr: CUdeviceptr,
    len: u64,
    /// Block returned from the block allocator
    block: Block,
    on_drop: Arc<Mutex<Vec<Block>>>,
}

impl Memory {
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    pub fn len(&self) -> u64 {
        self.len
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        self.on_drop.lock().push(self.block);
    }
}

/// Allocated page of CUDA memory.
struct Page {
    ptr: CUdeviceptr,
    start: u64,
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
                let memory = allocator.alloc(size, align).unwrap();
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
