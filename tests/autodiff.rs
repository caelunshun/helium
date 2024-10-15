use approx::assert_ulps_eq;
use helium::{AdTensor, Device, Param, Tensor};

const DEVICE: Device = Device::Cuda(0);

#[test]
fn autodiff_scalar() {
    let param1 = Param::new(Tensor::<1>::from_array([10.0], DEVICE));
    let param2 = Param::new(Tensor::<1>::from_array([8.0], DEVICE));

    let input = AdTensor::new(Tensor::<1>::from_array([2.0], DEVICE));

    let x = input * param1.clone();
    let x = x.pow_scalar(2.0);
    let x = x * param2.clone();

    let mut grads = x.backward();

    let grad_param1: Tensor<1> = grads.remove(param1.id());
    let grad_param2: Tensor<1> = grads.remove(param2.id());

    assert_ulps_eq!(grad_param1.into_scalar::<f32>(), 640.0);
    assert_ulps_eq!(grad_param2.into_scalar::<f32>(), 400.0);
}

#[test]
fn reduce_sum_mean() {
    let param = Param::new(Tensor::<1>::from_array([10.0f32; 10], DEVICE));
    let input = AdTensor::new(Tensor::<1>::from_array([2.0f32; 10], DEVICE));

    let result_sum = (input.clone() * param.clone()).reduce_sum::<1>(1) * 3.0;
    let result_mean = (input * param.clone()).reduce_mean::<1>(1) * 3.0;

    let backward_sum = result_sum.backward();
    assert_ulps_eq!(
        backward_sum
            .get::<1>(param.id())
            .clone()
            .into_vec::<f32>()
            .as_slice(),
        &[6.0f32; 10][..]
    );

    let backward_mean = result_mean.backward();
    assert_ulps_eq!(
        backward_mean
            .get::<1>(param.id())
            .clone()
            .into_vec::<f32>()
            .as_slice(),
        &[0.6f32; 10][..]
    );
}

#[test]
fn basic_gradient_descent() {
    // Use gradient descent to converge a parameter onto sqrt(X)
    const X: f32 = 1001.0;

    let mut guess = Param::new(Tensor::<1>::from_array([1.0], DEVICE));
    loop {
        let result = AdTensor::from(guess.clone()).pow_scalar(2.0);
        let error = (result - X).pow_scalar(2.0);
        let mut grads = error.clone().backward();

        let error = error.into_value().into_scalar::<f32>().sqrt();
        if error < 1e-3 {
            // Converged!
            println!(
                "{error} error with guess = {}",
                guess.value().clone().into_scalar::<f32>()
            );
            break;
        }

        println!(
            "error: {error}, guess: {}",
            guess.value().clone().into_scalar::<f32>()
        );
        let learning_rate = 1e-4;
        let gradient: Tensor<1> = grads.remove(guess.id());

        guess.set_value(guess.value().clone() - (gradient * learning_rate));
    }
}
