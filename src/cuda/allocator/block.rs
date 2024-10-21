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
            in_use_by_streams: BTreeSet::new(),
        };
        self.add_free_block(new_block);

        page
    }

    #[profiling::function]
    pub fn allocate(
        &mut self,
        size: u64,
        align: u64,
        allocating_stream: Option<StreamId>,
    ) -> Option<Block> {
        assert!(size > 0);
        assert!(align.is_power_of_two());

        for (_, size_class) in self.free_blocks_by_size.range(size..) {
            for &block_id in size_class {
                let block = &self.free_blocks[block_id];

                let matches_stream = match block.in_use_by_streams.len() {
                    0 => true,
                    1 => {
                        let stream = block.in_use_by_streams.iter().copied().next().unwrap();
                        match allocating_stream {
                            Some(allocating_stream) => {
                                stream == allocating_stream
                                    || self.streams[allocating_stream].ancestors.contains(&stream)
                            }
                            None => false,
                        }
                    }
                    _ => false,
                };

                if !matches_stream {
                    continue;
                }

                let offset = block.align_offset(align);
                if offset > block.size || (block.size - offset) < size {
                    continue;
                }

                let mut block = self.remove_free_block(block_id);

                if offset != 0 {
                    let (unused, new_block) = block.split(offset);
                    self.add_free_block(unused);
                    block = new_block;
                }

                let remainder = block.size - size;
                if remainder > 0 {
                    let size = block.size;
                    let (new_block, unused) = block.split(size - remainder);
                    self.add_free_block(unused);
                    block = new_block;
                }

                block.in_use_by_streams.extend(allocating_stream);

                return Some(block);
            }
        }

        None
    }

    /// Deallocates the given block.
    ///
    /// Behavior is unspecified if the block was not previously
    /// allocated from `self`, or if it was already freed.
    #[profiling::function]
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
            let adjacent = &self.free_blocks[adjacent_id];
            if block.can_merge(adjacent) {
                let adjacent = adjacent.clone();
                block = block.merge(self.free_blocks.remove(adjacent_id).unwrap());
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

                for &stream in &block.in_use_by_streams {
                    assert!(
                        self.streams[stream].blocks_in_use.remove(&adjacent_id),
                        "not present in blocks_in_use index"
                    );
                }
            }
        }

        let block_id = self.free_blocks.insert(block);
        let block = &mut self.free_blocks[block_id];
        self.free_blocks_by_size
            .entry(block.size)
            .or_default()
            .insert(block_id);
        page.free_blocks_by_addr.insert(block.start, block_id);

        block.in_use_by_streams.retain(|&stream| {
            if let Some(stream) = self.streams.get_mut(stream) {
                stream.blocks_in_use.insert(block_id);
                true
            } else {
                false
            }
        });
    }

    fn remove_free_block(&mut self, block_id: BlockId) -> Block {
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

        for &stream_id in &block.in_use_by_streams {
            if let Some(stream) = self.streams.get_mut(stream_id) {
                assert!(
                    stream.blocks_in_use.remove(&block_id),
                    "block not in blocks_in_use"
                );
            }
        }

        block
    }

    pub fn begin_stream(&mut self, parent: Option<StreamId>) -> StreamId {
        let ancestors = match parent {
            Some(p) if self.streams.contains_key(p) => self.streams[p]
                .ancestors
                .iter()
                .copied()
                .chain([p])
                .collect(),
            _ => BTreeSet::new(),
        };

        self.streams.insert(Stream {
            ancestors,
            ..Default::default()
        })
    }

    pub fn end_stream(&mut self, id: StreamId) {
        let stream = self.streams.remove(id).expect("stream already ended");
        for block in stream.blocks_in_use {
            let mut block = self.remove_free_block(block);
            block.in_use_by_streams.remove(&id);
            self.add_free_block(block);
        }

        for stream in self.streams.values_mut() {
            stream.ancestors.remove(&id);
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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Block {
    pub page: PageId,
    pub start: BlockAddr,
    pub size: BlockSize,
    /// Set of streams that may be currently using this block.
    ///
    /// A block can be allocated only if either
    /// (a) this set is empty, in which case the block is definitely
    /// not in use
    /// (b) this set has size 1, and the allocating stream matches
    /// the element of the set, in which case the block can be in
    /// use by the stream but will no longer be in use by the time
    /// the allocation is used
    in_use_by_streams: BTreeSet<StreamId>,
}

impl Block {
    pub fn is_aligned_to(&self, align: u64) -> bool {
        self.start % align == 0
    }

    pub fn align_offset(&self, align: u64) -> u64 {
        if self.is_aligned_to(align) {
            return 0;
        }
        align - (self.start % align)
    }

    pub fn end(&self) -> BlockAddr {
        self.start + self.size
    }

    pub fn can_merge(&self, other: &Self) -> bool {
        self.page == other.page
            && (self.start == other.end() || self.end() == other.start)
            && self.in_use_by_streams == other.in_use_by_streams
    }

    pub fn merge(self, other: Self) -> Self {
        debug_assert!(self.can_merge(&other));
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
    pub fn overlaps(&self, other: &Self) -> bool {
        (self.start >= other.start && self.start < other.end())
            || (other.start >= self.start && other.start < self.end())
    }

    pub fn split(self, at: u64) -> (Self, Self) {
        (
            Self {
                start: self.start,
                size: at,
                ..self.clone()
            },
            Self {
                start: self.start + at,
                size: self.size - at,
                ..self
            },
        )
    }

    pub fn mark_in_use_by_stream(&mut self, stream: StreamId) {
        self.in_use_by_streams.insert(stream);
    }
}

slotmap::new_key_type! {
    pub struct StreamId;
}

#[derive(Debug, Clone, Default)]
struct Stream {
    blocks_in_use: BTreeSet<BlockId>,
    /// List of (live) streams that happen-before this stream.
    ancestors: BTreeSet<StreamId>,
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

        let block1 = allocator.allocate(128, 256, None).unwrap();
        assert_eq!(block1.size, 128);

        let block2 = allocator.allocate(1024 - 128, 128, None).unwrap();
        assert_eq!(block2.size, 1024 - 128);

        assert!(allocator.allocate(1, 1, None).is_none());

        allocator.deallocate(block1);
        allocator.deallocate(block2);

        assert_eq!(allocator.allocate(1024, 8, None).unwrap().size, 1024);
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
                let block = allocator.allocate(size, 256, None).unwrap();
                assert_eq!(block.size, size);
                assert!(block.is_aligned_to(256));
                assert!(!allocated_blocks.iter().any(|b2| block.overlaps(b2)));
                allocated_blocks.push(block);
            } else {
                let i = rng.gen_range(0..allocated_blocks.len());
                let block = allocated_blocks.swap_remove(i);
                allocator.deallocate(block);
            }
        }
    }
}
