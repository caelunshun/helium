use std::{collections::BTreeMap, mem};

/// Allocator operating on a generalized address space,
/// represented by ranges of integers.
///
/// Allocation and deallocation are average-case O(log n) in the number
/// of blocks.
#[derive(Default)]
pub struct BlockAllocator {
    free_blocks_by_size: BTreeMap<BlockSize, Vec<Block>>,
    free_blocks_by_addr: BTreeMap<BlockAddr, Block>,
    end_address: u64,
    #[cfg(debug_assertions)]
    allocated_blocks: ahash::AHashSet<Block>,
    page_size: u64,
}

impl BlockAllocator {
    pub fn new(page_size: u64) -> Self {
        assert!(page_size > 0);
        Self {
            page_size,
            ..Default::default()
        }
    }

    /// Allocates a block of the given size and alignment, returning
    /// the block address and size;
    pub fn alloc(&mut self, size: u64, align: u64) -> Block {
        assert!(size > 0);
        assert!(align.is_power_of_two());

        // Try to find the smallest block satisfying the constraints;
        // if none found, then we add more used space.
        let mut valid_block = None;
        for block in self.free_blocks_by_size.range(size..).flat_map(|(_, b)| b) {
            let align_offset = align_offset(block.addr, align);
            let Some(effective_size) = block.size.checked_sub(align_offset) else {
                continue;
            };
            if effective_size >= size {
                valid_block = Some((*block, align_offset));
                break;
            }
        }

        let (mut block, align_offset) = match valid_block {
            Some(b) => b,
            None => return self.alloc_at_end(size, align),
        };

        self.remove_free_block(block);

        if align_offset > 0 {
            let (unused, new_block) = block.split(align_offset);
            self.add_free_block(unused);
            block = new_block;
        }

        if block.size > size {
            let (new_block, unused) = block.split(size);
            self.add_free_block(unused);
            block = new_block;
        }

        #[cfg(debug_assertions)]
        assert!(!self.allocated_blocks.iter().any(|b| b.overlaps(block)));
        #[cfg(debug_assertions)]
        self.allocated_blocks.insert(block);

        block
    }

    fn alloc_at_end(&mut self, size: u64, align: u64) -> Block {
        let fits_in_page = ((self.end_address + size) / self.page_size
            == self.end_address / self.page_size)
            || self.end_address % self.page_size == 0;
        if !fits_in_page {
            let align_offset = align_offset(self.end_address, self.page_size);
            self.add_free_block(Block {
                addr: self.end_address,
                size: align_offset,
            });
            self.end_address += align_offset;
        }

        let align_offset = align_offset(self.end_address, align);
        if align_offset > 0 {
            self.add_free_block(Block {
                addr: self.end_address,
                size: align_offset,
            });
            self.end_address += align_offset;
        }

        let addr = self.end_address;
        self.end_address += size;

        let block = Block { addr, size };

        #[cfg(debug_assertions)]
        self.allocated_blocks.insert(block);

        block
    }

    /// Frees the space occupied by a block, allowing its range
    /// to be reused.
    ///
    /// Behavior is unspecified if `block` was not returned
    /// from a previous call to `self.alloc()` or if it was
    /// already deallocated.
    pub fn dealloc(&mut self, block: Block) {
        #[cfg(debug_assertions)]
        assert!(
            self.allocated_blocks.remove(&block),
            "called dealloc() with a block that was not allocated (e.g., double-free)"
        );

        self.add_free_block(block);
    }

    fn can_merge(&self, a: Block, b: Block) -> bool {
        let adjacent = a.is_adjacent_to(b);
        let same_page = a.addr / self.page_size == b.addr / self.page_size;
        adjacent && same_page
    }

    fn add_free_block(&mut self, mut block: Block) {
        // Merge with adjacent free blocks.
        let next_block = self
            .free_blocks_by_addr
            .range(block.addr + 1..)
            .map(|(_, b)| *b)
            .next();
        if let Some(next_block) = next_block {
            if self.can_merge(block, next_block) {
                self.remove_free_block(next_block);
                block = block.merge_with(next_block);
            }
        }

        let prev_block = self
            .free_blocks_by_addr
            .range(..block.addr)
            .rev()
            .map(|(_, b)| *b)
            .next();
        if let Some(prev_block) = prev_block {
            if self.can_merge(block, prev_block) {
                self.remove_free_block(prev_block);
                block = block.merge_with(prev_block);
            }
        }

        self.free_blocks_by_size
            .entry(block.size)
            .or_default()
            .push(block);
        self.free_blocks_by_addr.insert(block.addr, block);
    }

    fn remove_free_block(&mut self, block: Block) {
        self.free_blocks_by_addr.remove(&block.addr);
        let list = self.free_blocks_by_size.get_mut(&block.size).unwrap();
        let pos = list
            .iter()
            .position(|b| b.addr == block.addr)
            .expect("block not in free list");
        list.swap_remove(pos);
    }
}

fn align_offset(addr: u64, align: u64) -> u64 {
    align - (addr % align)
}

type BlockAddr = u64;
type BlockSize = u64;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Block {
    pub addr: BlockAddr,
    pub size: BlockSize,
}

impl Block {
    pub fn is_adjacent_to(&self, other: Block) -> bool {
        self.addr + self.size == other.addr || other.addr + other.size == self.addr
    }

    #[allow(unused)]
    pub fn overlaps(mut self, mut other: Block) -> bool {
        if other.addr < self.addr {
            mem::swap(&mut self, &mut other);
        }
        other.addr >= self.addr && other.addr < self.addr + self.size
    }

    #[must_use]
    pub fn merge_with(self, other: Block) -> Block {
        debug_assert!(self.is_adjacent_to(other));
        let addr = self.addr.min(other.addr);
        let size = self.size + other.size;
        Block { addr, size }
    }

    #[must_use]
    pub fn split(self, offset: u64) -> (Block, Block) {
        debug_assert!(offset > 0 && offset < self.size);
        (
            Block {
                addr: self.addr,
                size: offset,
            },
            Block {
                addr: self.addr + offset,
                size: self.size - offset,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_allocator() {
        let allocator = BlockAllocator::new(1024);
        assert_eq!(allocator.end_address, 0);
    }

    #[test]
    fn test_simple_alloc_and_dealloc() {
        let mut allocator = BlockAllocator::new(1024);
        let block = allocator.alloc(100, 8);
        assert_eq!(block.size, 100);
        assert_eq!(block.addr % 8, 0);
        allocator.dealloc(block);
    }

    #[test]
    fn test_multiple_allocs() {
        let mut allocator = BlockAllocator::new(1024);
        let block1 = allocator.alloc(50, 8);
        let block2 = allocator.alloc(100, 16);
        let block3 = allocator.alloc(75, 32);

        assert_eq!(block1.size, 50);
        assert_eq!(block2.size, 100);
        assert_eq!(block3.size, 75);

        assert_eq!(block1.addr % 8, 0);
        assert_eq!(block2.addr % 16, 0);
        assert_eq!(block3.addr % 32, 0);

        allocator.dealloc(block1);
        allocator.dealloc(block2);
        allocator.dealloc(block3);
    }

    #[test]
    fn test_reuse_freed_block() {
        let mut allocator = BlockAllocator::new(1024);
        let block1 = allocator.alloc(100, 8);
        allocator.dealloc(block1);
        let block2 = allocator.alloc(50, 8);
        assert_eq!(block1.addr, block2.addr);
    }

    #[test]
    fn test_merge_adjacent_blocks() {
        let mut allocator = BlockAllocator::new(1024);
        let block1 = allocator.alloc(100, 8);
        let block2 = allocator.alloc(100, 8);
        let block3 = allocator.alloc(100, 8);

        allocator.dealloc(block1);
        allocator.dealloc(block3);
        allocator.dealloc(block2);

        let large_block = allocator.alloc(300, 8);
        assert_eq!(large_block.size, 300);
        assert_eq!(large_block.addr, block1.addr);
    }

    #[test]
    fn test_alignment() {
        let mut allocator = BlockAllocator::new(1024);
        let _padding = allocator.alloc(1, 4);
        let block = allocator.alloc(100, 64);
        assert_eq!(block.addr % 64, 0);
    }
}
