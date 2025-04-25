use crate::cuda::{
    allocator::block::{Block, BlockAllocator, PageId},
    context::CudaContext,
    error::CudaError,
};
use cudarc::driver::sys::{cuCtxSetCurrent, cuMemHostAlloc, CUcontext, CU_MEMHOSTALLOC_DEVICEMAP};
use parking_lot::{Mutex, MutexGuard};
use slotmap::SecondaryMap;
use std::{
    alloc::Layout,
    ptr,
    ptr::NonNull,
    sync::{Arc, OnceLock},
};

/// Allocator for page-locked host memory,
/// which is used as the host endpoint
/// for all device <=> host memory transfers.
///
/// Underlying memory is allocated with cudaHostAlloc(),
/// but we cache these allocations to improve performance.
pub struct HostPinnedAllocator {
    cx: CUcontext,
    block_allocator: BlockAllocator,
    pages: SecondaryMap<PageId, Page>,
    dropped_memories: Arc<Mutex<Vec<Block>>>,
}

unsafe impl Send for HostPinnedAllocator {}
unsafe impl Sync for HostPinnedAllocator {}

impl HostPinnedAllocator {
    pub const MAX_ALIGN: usize = 256;
    const MIN_PAGE_SIZE: usize = 128 * 1024 * 1024;

    /// Gets the global host-pinned memory allocator.
    pub fn global() -> MutexGuard<'static, Self> {
        static GLOBAL_ALLOCATOR: OnceLock<Mutex<HostPinnedAllocator>> = OnceLock::new();
        GLOBAL_ALLOCATOR
            .get_or_init(|| Mutex::new(Self::new(CudaContext::global(0).unwrap().raw_context())))
            .lock()
    }

    /// # Safety
    /// `cx` must outlive `self`.
    pub fn new(cx: CUcontext) -> Self {
        Self {
            cx,
            block_allocator: BlockAllocator::new(),
            pages: SecondaryMap::default(),
            dropped_memories: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Allocates some host-pinned memory.
    pub fn alloc(&mut self, layout: Layout) -> Result<HostPinnedMemory, CudaError> {
        assert!(layout.align() <= Self::MAX_ALIGN);
        assert!(layout.align().is_power_of_two());

        self.process_dropped_memories();

        let block =
            match self
                .block_allocator
                .allocate(layout.size() as u64, layout.align() as u64, None)
            {
                Some(block) => block,
                None => {
                    self.new_page_for_at_least(layout.size())?;
                    self.block_allocator
                        .allocate(layout.size() as u64, layout.align() as u64, None)
                        .unwrap()
                }
            };

        let ptr = unsafe { self.pages[block.page].base.add(block.start as usize) };

        Ok(HostPinnedMemory {
            block,
            ptr,
            on_drop: Arc::clone(&self.dropped_memories),
        })
    }

    fn process_dropped_memories(&mut self) {
        for block in self.dropped_memories.lock().drain(..) {
            self.block_allocator.deallocate(block);
        }
    }

    fn new_page_for_at_least(&mut self, min_size: usize) -> Result<(), CudaError> {
        unsafe {
            cuCtxSetCurrent(self.cx).result()?;
        }

        let size = Self::MIN_PAGE_SIZE.max(min_size).next_power_of_two();

        let mut ptr = ptr::null_mut();

        unsafe {
            cuMemHostAlloc(&mut ptr, size, CU_MEMHOSTALLOC_DEVICEMAP)
                .result()
                .unwrap()
        };

        let page_id = self.block_allocator.add_page(size as u64);
        self.pages.insert(
            page_id,
            Page {
                base: NonNull::new(ptr).unwrap().cast(),
            },
        );

        Ok(())
    }
}

struct Page {
    base: NonNull<u8>,
}

unsafe impl Send for Page {}
unsafe impl Sync for Page {}

/// Handle to a block of host-pinned memory.
///
/// Freed on drop.
pub struct HostPinnedMemory {
    ptr: NonNull<u8>,
    block: Block,
    on_drop: Arc<Mutex<Vec<Block>>>,
}

impl HostPinnedMemory {
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

unsafe impl Send for HostPinnedMemory {}
unsafe impl Sync for HostPinnedMemory {}

impl Drop for HostPinnedMemory {
    fn drop(&mut self) {
        self.on_drop.lock().push(self.block.clone());
    }
}
