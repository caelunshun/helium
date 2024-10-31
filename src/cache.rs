use lru::LruCache;
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use std::{hash::Hash, num::NonZeroUsize};

/// Thread-safe LRU cache wrapper. Intended for use in `static`s.
pub struct Cache<K, V> {
    lru: Mutex<Option<LruCache<K, V, ahash::RandomState>>>,
    capacity: NonZeroUsize,
}

impl<K, V> Cache<K, V>
where
    K: Eq + Hash,
{
    pub const fn with_capacity(capacity: usize) -> Self {
        Self {
            lru: Mutex::new(None),
            capacity: NonZeroUsize::new(capacity).unwrap(),
        }
    }

    fn write(&self) -> MappedMutexGuard<LruCache<K, V, ahash::RandomState>> {
        let guard = self.lru.lock();
        MutexGuard::map(guard, |opt| match opt {
            Some(lru) => lru,
            None => opt.insert(LruCache::with_hasher(
                self.capacity,
                ahash::RandomState::new(),
            )),
        })
    }

    pub fn get_or_insert(&self, key: &K, insert_with: impl FnOnce() -> V) -> V
    where
        V: Clone,
        K: Clone,
    {
        let mut guard = self.write();
        if let Some(val) = guard.get(key) {
            return val.clone();
        }

        // Drop guard while initializing to allow other threads
        // to make progress. Note: this introduces the possibility
        // of redundant initialization, which is an acceptable tradeoff.
        drop(guard);

        let val = insert_with();
        self.write().get_or_insert(key.clone(), move || val).clone()
    }
}
