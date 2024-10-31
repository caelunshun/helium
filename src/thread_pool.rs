#![allow(unused)]

use flume::Sender;
use std::{sync::OnceLock, thread};

/// Simple thread pool designed for offloading blocking
/// operations, e.g. CUDA sync calls.
pub struct BlockingThreadPool {
    num_threads: usize,
    /// Lazily initialized on first use.
    task_sender: OnceLock<Sender<Task>>,
}

type Task = Box<dyn FnOnce() + Send + 'static>;

impl BlockingThreadPool {
    pub const fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            task_sender: OnceLock::new(),
        }
    }

    /// Gets the global blocking thread pool.
    pub fn global() -> &'static Self {
        static POOL: BlockingThreadPool = BlockingThreadPool::new(32);
        &POOL
    }

    pub fn spawn<F>(&self, task: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.task_sender
            .get_or_init(|| {
                let (sender, receiver) = flume::bounded::<Task>(128);
                for i in 0..self.num_threads {
                    let receiver = receiver.clone();
                    thread::Builder::new()
                        .name(format!("blocking-thread-{i}"))
                        .spawn(move || {
                            for task in receiver {
                                task();
                            }
                        })
                        .expect("failed to spawn thread");
                }
                sender
            })
            .send(Box::new(task))
            .unwrap();
    }
}
