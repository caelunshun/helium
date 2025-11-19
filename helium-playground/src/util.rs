use std::hash::{BuildHasher, Hash, Hasher};

pub mod kernel_builder;

pub fn order_independent_hash<T: Hash>(
    iter: impl IntoIterator<Item = T>,
    hasher: &mut impl Hasher,
) {
    let mut hash = 0u64;

    let temp_hasher = foldhash::quality::FixedState::with_seed(14425774135239826768);
    for item in iter {
        hash ^= temp_hasher.hash_one(item);
    }

    hash.hash(hasher);
}

pub fn order_independent_hash_one<T: Hash>(
    iter: impl IntoIterator<Item = T>,
    builder: impl BuildHasher,
) -> u64 {
    let mut hasher = builder.build_hasher();
    order_independent_hash(iter, &mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_independence() {
        let hash_builder = foldhash::quality::FixedState::with_seed(1);
        assert_eq!(
            order_independent_hash_one([1, 2, 3, 4, 5, 6], hash_builder.clone()),
            order_independent_hash_one([6, 5, 4, 3, 2, 1], hash_builder),
        );
    }
}
