use helium::{Device, Param, Tensor};
use pollster::FutureExt;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rand_pcg::Pcg64Mcg;
use std::{future::Future, pin::Pin, thread};

const DEVICE: Device = Device::Cuda(0);

#[cfg_attr(debug_assertions, ignore)]
#[test]
fn determinism_stress_test() {
    tracing_subscriber::fmt::init();
    thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                let mut rng = Pcg64Mcg::seed_from_u64(783542);

                let d = 1024;

                let a1 = Param::new(random_tensor(&mut rng, [d, d]));
                let a2 = random_tensor(&mut rng, [d, d]);

                let x = random_tensor(&mut rng, [d]);

                let expected = do_ops(a1.value(), &a2, &x);
                let grads = expected.reduce_mean::<1>(1).backward();
                let expected_grad = grads.get::<2>(a1.id()).to_vec::<f32>();
                let expected = expected.to_vec::<f32>();

                let (result_tx, result_rx) = flume::bounded::<
                    Pin<Box<dyn Future<Output = (Vec<f32>, Vec<f32>)> + Send + Sync>>,
                >(4);
                let thread = thread::spawn(move || {
                    for result in result_rx {
                        let (result, grad) = result.block_on();
                        assert_eq!(result, expected);
                        assert_eq!(grad, expected_grad);
                    }
                });

                for i in 0..100_000 {
                    if i % 1000 == 0 {
                        dbg!(i);
                    }
                    let result = do_ops(a1.value(), &a2, &x);
                    let grad = result.reduce_mean::<1>(1).backward();
                    let grad = grad.get::<2>(a1.id());

                    grad.async_start_eval();

                    result_tx
                        .send(Box::pin(async move {
                            let grad = grad.to_vec_async::<f32>().await;
                            let result = result.to_vec_async::<f32>().await;
                            (result, grad)
                        }))
                        .unwrap();
                }

                drop(result_tx);

                thread.join().unwrap();
            });
        }
    });
}

fn do_ops(a1: &Tensor<2>, a2: &Tensor<2>, x: &Tensor<1>) -> Tensor<1> {
    let y = a1.matmul(x.reshape([x.shape()[0], 1])).sigmoid();
    let y = y
        .reduce_sum::<2>(1)
        .broadcast_to([x.shape()[0], x.shape()[0]]);
    let y = a2.matmul(y).cos() + 1.0;
    y.reduce_mean::<2>(1).reshape([x.shape()[0]])
}

fn random_tensor<const D: usize>(rng: &mut impl Rng, shape: [usize; D]) -> Tensor<D> {
    let data = (0..shape.iter().copied().product::<usize>())
        .map(|_| StandardNormal.sample(rng))
        .map(|x: f32| x * 0.1)
        .collect::<Vec<f32>>();
    Tensor::from_slice(data, shape, DEVICE)
}
