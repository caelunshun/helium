use slotmap::SlotMap;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Default)]
pub struct BlockAllocator {
    pages: SlotMap<PageId, Page>,
    free_blocks: SlotMap<BlockId, Block>,
    streams: SlotMap<StreamId, Stream>,
    /// Acceleration structure storing all blocks
    /// across all pages, ordered by size.
    free_blocks_by_size: BTreeMap<BlockSize, BTreeSet<BlockId>>,
}

impl BlockAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a new page with the given size.
    pub fn add_page(&mut self, size: u64) -> PageId {
        let page = self.pages.insert(Page::default());

        let new_block = Block {
            page,
            start: 0,
            size,
            free_only_in_stream: None,
        };
        self.add_free_block(new_block);

        page
    }

    pub fn allocate(&mut self, size: u64, align: u64) -> Option<Block> {
        self.allocate_with_opt_stream(size, align, None)
    }

    #[allow(unused)]
    pub fn allocate_in_stream(&mut self, size: u64, align: u64, stream: StreamId) -> Option<Block> {
        self.allocate_with_opt_stream(size, align, Some(stream))
    }

    pub fn allocate_with_opt_stream(
        &mut self,
        size: u64,
        align: u64,
        stream: Option<StreamId>,
    ) -> Option<Block> {
        assert!(size > 0);
        assert!(align.is_power_of_two());

        for (_, size_class) in self.free_blocks_by_size.range(size..) {
            for &block_id in size_class {
                let mut block = self.free_blocks[block_id];

                if block.free_only_in_stream.is_some() && block.free_only_in_stream != stream {
                    continue;
                }

                let offset = block.align_offset(align);
                if offset > block.size || (block.size - offset) < size {
                    continue;
                }

                self.remove_free_block(block_id);

                if offset != 0 {
                    let (unused, new_block) = block.split(offset);
                    self.add_free_block(unused);
                    block = new_block;
                }

                let remainder = block.size - size;
                if remainder > 0 {
                    let (new_block, unused) = block.split(block.size - remainder);
                    self.add_free_block(unused);
                    block = new_block;
                }

                block.free_only_in_stream = stream;
                return Some(block);
            }
        }

        None
    }

    /// Deallocates the given block.
    ///
    /// Behavior is unspecified if the block was not previously
    /// allocated from `self`, or if it was already freed.
    pub fn deallocate(&mut self, block: Block) {
        self.add_free_block(block);
    }

    fn add_free_block(&mut self, mut block: Block) {
        // Try to merge the block with adjacent blocks.
        let page = &mut self.pages[block.page];
        let previous = page
            .free_blocks_by_addr
            .range(..block.start)
            .last()
            .map(|(_, &b)| b);
        let next = page
            .free_blocks_by_addr
            .range(block.start + 1..)
            .next()
            .map(|(_, &b)| b);

        for adjacent_id in [previous, next].into_iter().flatten() {
            let adjacent = self.free_blocks[adjacent_id];
            if block.can_merge(adjacent) {
                block = block.merge(adjacent);
                self.free_blocks.remove(adjacent_id);
                assert!(
                    self.free_blocks_by_size
                        .get_mut(&adjacent.size)
                        .unwrap()
                        .remove(&adjacent_id),
                    "not present in size index"
                );
                page.free_blocks_by_addr
                    .remove(&adjacent.start)
                    .expect("not present in addr index");

                if let Some(stream) = adjacent.free_only_in_stream {
                    assert!(
                        self.streams[stream]
                            .blocks_free_only_in_stream
                            .remove(&adjacent_id),
                        "not present in free_only_in_stream index"
                    );
                }
            }
        }

        let block_id = self.free_blocks.insert(block);
        self.free_blocks_by_size
            .entry(block.size)
            .or_default()
            .insert(block_id);
        page.free_blocks_by_addr.insert(block.start, block_id);

        if let Some(stream) = block.free_only_in_stream {
            if let Some(stream) = self.streams.get_mut(stream) {
                stream.blocks_free_only_in_stream.insert(block_id);
            } else {
                self.free_blocks[block_id].free_only_in_stream = None;
            }
        }
    }

    fn remove_free_block(&mut self, block_id: BlockId) {
        let block = self
            .free_blocks
            .remove(block_id)
            .expect("block already removed");
        self.pages[block.page]
            .free_blocks_by_addr
            .remove(&block.start)
            .expect("block not in free_blocks_by_addr");
        assert!(
            self.free_blocks_by_size
                .get_mut(&block.size)
                .unwrap()
                .remove(&block_id),
            "block not in free_blocks_by_size"
        );

        if self.free_blocks_by_size[&block.size].is_empty() {
            self.free_blocks_by_size.remove(&block.size);
        }

        if let Some(stream) = block.free_only_in_stream {
            if let Some(stream) = self.streams.get_mut(stream) {
                assert!(
                    stream.blocks_free_only_in_stream.remove(&block_id),
                    "block not in free_only_in_stream"
                );
            }
        }
    }

    pub fn begin_stream(&mut self) -> StreamId {
        self.streams.insert(Stream::default())
    }

    pub fn end_stream(&mut self, id: StreamId) {
        let stream = self.streams.remove(id).expect("stream already ended");
        for block in stream.blocks_free_only_in_stream {
            let modified_block = Block {
                free_only_in_stream: None,
                ..self.free_blocks[block]
            };
            self.remove_free_block(block);
            self.add_free_block(modified_block);
        }
    }
}

slotmap::new_key_type! {
    pub struct PageId;
}

#[derive(Debug, Clone, Default)]
struct Page {
    /// Free blocks in this page, ordered by start
    /// address. Used to accelerate block coalescing.
    free_blocks_by_addr: BTreeMap<BlockAddr, BlockId>,
}

type BlockAddr = u64;
type BlockSize = u64;

slotmap::new_key_type! {
     struct BlockId;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Block {
    pub page: PageId,
    pub start: BlockAddr,
    pub size: BlockSize,
    /// If the block was previously allocated within a stream,
    /// then freed, and the stream is still alive,
    /// this is the ID of that stream.
    free_only_in_stream: Option<StreamId>,
}

impl Block {
    pub fn is_aligned_to(self, align: u64) -> bool {
        self.start % align == 0
    }

    pub fn align_offset(self, align: u64) -> u64 {
        if self.is_aligned_to(align) {
            return 0;
        }
        align - (self.start % align)
    }

    pub fn end(self) -> BlockAddr {
        self.start + self.size
    }

    pub fn can_merge(self, other: Self) -> bool {
        self.page == other.page
            && (self.start == other.end() || self.end() == other.start)
            && self.free_only_in_stream == other.free_only_in_stream
    }

    pub fn merge(self, other: Self) -> Self {
        debug_assert!(self.can_merge(other));
        let start = self.start.min(other.start);
        let end = self.end().max(other.end());
        let size = end - start;
        Self {
            start,
            size,
            ..self
        }
    }

    #[allow(unused)]
    pub fn overlaps(self, other: Self) -> bool {
        (self.start >= other.start && self.start < other.end())
            || (other.start >= self.start && other.start < self.end())
    }

    pub fn split(self, at: u64) -> (Self, Self) {
        (
            Self {
                start: self.start,
                size: at,
                ..self
            },
            Self {
                start: self.start + at,
                size: self.size - at,
                ..self
            },
        )
    }
}

slotmap::new_key_type! {
    pub struct StreamId;
}

#[derive(Debug, Clone, Default)]
struct Stream {
    /// List of blocks having `free_only_in_stream` set to this stream.
    blocks_free_only_in_stream: BTreeSet<BlockId>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    #[test]
    fn simple() {
        let mut allocator = BlockAllocator::new();
        allocator.add_page(1024);

        let block1 = allocator.allocate(128, 256).unwrap();
        assert_eq!(block1.size, 128);

        let block2 = allocator.allocate(1024 - 128, 128).unwrap();
        assert_eq!(block2.size, 1024 - 128);

        assert!(allocator.allocate(1, 1).is_none());

        allocator.deallocate(block1);
        allocator.deallocate(block2);

        assert_eq!(allocator.allocate(1024, 8).unwrap().size, 1024);
    }

    #[test]
    fn stress_test() {
        let mut allocator = BlockAllocator::new();
        allocator.add_page(64 * 1024 * 1024 * 1024);

        let mut allocated_blocks: Vec<Block> = Vec::new();
        let mut rng = Pcg64Mcg::seed_from_u64(66);
        for _ in 0..25_000 {
            if allocated_blocks.is_empty() || rng.gen_bool(0.4) {
                let size = rng.gen_range(1024..1024 * 1024 * 32);
                let block = allocator.allocate(size, 256).unwrap();
                assert_eq!(block.size, size);
                assert!(block.is_aligned_to(256));
                assert!(!allocated_blocks.iter().any(|b2| block.overlaps(*b2)));
                allocated_blocks.push(block);
            } else {
                let i = rng.gen_range(0..allocated_blocks.len());
                let block = allocated_blocks.swap_remove(i);
                allocator.deallocate(block);
            }
        }
    }
}
