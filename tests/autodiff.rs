use approx::assert_ulps_eq;
use helium::{Device, Param, Tensor};

const DEVICE: Device = Device::Cuda(0);

#[test]
fn simple() {
    let param = Param::new(Tensor::from_scalar(2.0, DEVICE));
    let result = param.value() * 3.0;

    let grads = result.backward();
    assert_eq!(grads.get::<1>(param.id()).to_scalar::<f32>(), 3.0);
}

#[test]
fn autodiff_scalar() {
    let param1 = Param::new(Tensor::<1>::from_array([10.0], DEVICE));
    let param2 = Param::new(Tensor::<1>::from_array([8.0], DEVICE));

    let input = Tensor::<1>::from_array([2.0], DEVICE);

    let x = input * param1.clone();
    let x = x.pow_scalar(2.0);
    let x = x * param2.clone();

    let grads = x.backward();

    let grad_param1: Tensor<1> = grads.get(param1.id());
    let grad_param2: Tensor<1> = grads.get(param2.id());

    assert_ulps_eq!(grad_param1.to_scalar::<f32>(), 640.0);
    assert_ulps_eq!(grad_param2.to_scalar::<f32>(), 400.0);
}

#[test]
fn reduce_sum_mean() {
    let param = Param::new(Tensor::<1>::from_array([10.0f32; 10], DEVICE));
    let input = Tensor::<1>::from_array([2.0f32; 10], DEVICE);

    let result_sum = (input.clone() * param.clone()).reduce_sum::<1>(1) * 3.0;
    let result_mean = (input * param.clone()).reduce_mean::<1>(1) * 3.0;

    let backward_sum = result_sum.backward();
    assert_ulps_eq!(
        backward_sum
            .get::<1>(param.id())
            .clone()
            .to_vec::<f32>()
            .as_slice(),
        &[6.0f32; 10][..]
    );

    let backward_mean = result_mean.backward();
    assert_ulps_eq!(
        backward_mean
            .get::<1>(param.id())
            .clone()
            .to_vec::<f32>()
            .as_slice(),
        &[0.6f32; 10][..]
    );
}

#[test]
fn broadcast() {
    let param = Param::new(Tensor::<2>::from_slice(vec![1.0f32; 20], [20, 1], DEVICE));

    let x = Tensor::<3>::from_slice(vec![2.0f32; 80], [2, 20, 2], DEVICE);

    let y = (param.value().broadcast_to(x.shape()) * x).reduce_sum::<1>(3);

    let grads = y.backward();
    let grad = grads.get::<2>(param.id());
    assert_ulps_eq!(grad.clone().to_vec::<f32>().as_slice(), &[8.0f32; 20][..]);
}

#[test]
fn reduce_min() {
    // Gradient should be evenly distributed among all elements
    // that equal the minimum.
    let param = Param::new(Tensor::<1>::from_array([1.0; 6], DEVICE));

    let x = Tensor::<1>::from_array([0.5, 2.0, 3.0, 3.0, 2.0, 0.5], DEVICE);

    let y = (param.value().broadcast_to(x.shape()) * x).reduce_min::<1>(1);

    let grads = y.backward();
    let grad = grads.get::<1>(param.id());

    assert_ulps_eq!(
        grad.to_vec::<f32>().as_slice(),
        &[0.25, 0.0, 0.0, 0.0, 0.0, 0.25][..]
    );
}

#[test]
fn reduce_max() {
    // Gradient should be evenly distributed among all elements
    // that equal the maximum.
    let param = Param::new(Tensor::<1>::from_array([1.0; 6], DEVICE));

    let x = Tensor::<1>::from_array([0.5, 2.0, 3.0, 3.0, 2.0, 0.5], DEVICE);

    let y = (param.value().broadcast_to(x.shape()) * x).reduce_max::<1>(1);

    let grads = y.backward();
    let grad = grads.get::<1>(param.id());

    assert_ulps_eq!(
        grad.to_vec::<f32>().as_slice(),
        &[0.0, 0.0, 1.5, 1.5, 0.0, 0.0][..]
    );
}

#[test]
fn basic_gradient_descent() {
    // Use gradient descent to converge a parameter onto sqrt(X)
    const X: f32 = 1001.0;

    let mut guess = Param::new(Tensor::<1>::from_array([1.0], DEVICE));
    loop {
        let result = guess.value().pow_scalar(2.0);
        let error = (result - X).pow_scalar(2.0);
        let grads = error.clone().backward();

        let error = error.to_scalar::<f32>().sqrt();
        if error < 1e-3 {
            // Converged!
            println!(
                "{error} error with guess = {}",
                guess.value().clone().to_scalar::<f32>()
            );
            break;
        }

        println!(
            "error: {error}, guess: {}",
            guess.value().clone().to_scalar::<f32>()
        );
        let learning_rate = 1e-4;
        let gradient: Tensor<1> = grads.get(guess.id());

        guess.set_value(guess.value().clone() - (gradient * learning_rate));
    }
}

#[test]
fn relu() {
    let p = Param::new(Tensor::<1>::from_array([1.0, 2.0, 3.0, -4.0], DEVICE));
    let x = Tensor::<1>::from_array([1.0, -1.0, 5.0, -10.0], DEVICE);
    let y = (&x * &p).relu();
    let grad: Tensor<1> = y.reduce_sum::<1>(1).backward().get(p.id());

    assert_ulps_eq!(grad.to_vec::<f32>().as_slice(), &[1.0, 0.0, 5.0, -10.0][..]);
}

#[test]
fn max_a_gt_b() {
    let a = Param::new(Tensor::from_constant(1.5f32, [1], DEVICE));
    let b = Param::new(Tensor::from_constant(0.5f32, [1], DEVICE));
    let result = a.value().max(&b) * 2.0;
    let grads = result.backward();

    assert_eq!(grads.get(a.id()).to_scalar::<f32>(), 2.0);
    assert_eq!(grads.get(b.id()).to_scalar::<f32>(), 0.0);
}

#[test]
fn max_b_gt_a() {
    let a = Param::new(Tensor::from_constant(-0.5f32, [1], DEVICE));
    let b = Param::new(Tensor::from_constant(2.5f32, [1], DEVICE));
    let result = a.value().max(&b) * 2.0;
    let grads = result.backward();

    assert_eq!(grads.get(a.id()).to_scalar::<f32>(), 0.0);
    assert_eq!(grads.get(b.id()).to_scalar::<f32>(), 2.0);
}

#[test]
fn max_ab_equal() {
    let a = Param::new(Tensor::from_constant(-0.5f32, [1], DEVICE));
    let b = Param::new(Tensor::from_constant(-0.5f32, [1], DEVICE));
    let result = a.value().max(&b) * 2.0;
    let grads = result.backward();

    assert_eq!(grads.get(a.id()).to_scalar::<f32>(), 2.0 * 0.5);
    assert_eq!(grads.get(b.id()).to_scalar::<f32>(), 2.0 * 0.5);
}

#[test]
fn min_a_lt_b() {
    let a = Param::new(Tensor::from_constant(-1.5f32, [1], DEVICE));
    let b = Param::new(Tensor::from_constant(0.5f32, [1], DEVICE));
    let result = a.value().min(&b) * 2.0;
    let grads = result.backward();

    assert_eq!(grads.get(a.id()).to_scalar::<f32>(), 2.0);
    assert_eq!(grads.get(b.id()).to_scalar::<f32>(), 0.0);
}

#[test]
fn min_b_lt_a() {
    let a = Param::new(Tensor::from_constant(0.5f32, [1], DEVICE));
    let b = Param::new(Tensor::from_constant(-2.5f32, [1], DEVICE));
    let result = a.value().min(&b) * 2.0;
    let grads = result.backward();

    assert_eq!(grads.get(a.id()).to_scalar::<f32>(), 0.0);
    assert_eq!(grads.get(b.id()).to_scalar::<f32>(), 2.0);
}

#[test]
fn min_ab_equal() {
    let a = Param::new(Tensor::from_constant(-0.5f32, [1], DEVICE));
    let b = Param::new(Tensor::from_constant(-0.5f32, [1], DEVICE));
    let result = a.value().min(&b) * 2.0;
    let grads = result.backward();

    assert_eq!(grads.get(a.id()).to_scalar::<f32>(), 2.0 * 0.5);
    assert_eq!(grads.get(b.id()).to_scalar::<f32>(), 2.0 * 0.5);
}
