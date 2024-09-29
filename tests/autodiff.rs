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
