//! Benchmark for `permute_dim` specialized kernel.
//!
//! Relies on blocking behavior of `async_start_eval`
//! when using `cuda-tracing` feature flag (required).

use helium::{Device, Tensor};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt::init();
    let device = Device::Cuda(0);

    let n = 2048;
    let h = 128;
    let w = 128;
    let c = 32;
    let data = vec![10.0f32; n * h * w * c];

    let x = Tensor::<4>::from_vec(data, [n, h, w, c], device);

    tracing::info!("Warmup");
    for _ in 0..5 {
        let y = x.swap_dims(0, 3);
        y.async_start_eval();
    }

    let start = Instant::now();
    let num_iter = 50;
    for _ in 0..num_iter {
        let y = x.swap_dims(0, 3);
        y.async_start_eval();
    }
    let time = start.elapsed() / num_iter;
    let bandwidth = (2 * size_of::<f32>() * n * h * w * c) as f64
        / time.as_secs_f64()
        / (1024.0 * 1024.0 * 1024.0);
    tracing::info!(
        "{:.2?}/iter, {:.2} GiB/s achieved bandwidth",
        time,
        bandwidth
    );
}
