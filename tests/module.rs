use helium::{
    module::{Module, ParamVisitor},
    Device, Param, ParamId, Tensor,
};
use helium_macros::Module;

#[test]
fn derive_module() {
    #[derive(Module)]
    struct M {
        param: Param<2>,
        #[module(config)]
        _config: u32,
    }

    let module = M {
        param: Param::new(Tensor::<2>::from_array([[1.0f32]], Device::Cuda(0))),
        _config: 4,
    };

    struct MockParamVisitor {
        expected_id: ParamId,
        found: bool,
    }

    impl ParamVisitor for MockParamVisitor {
        fn visit_param<const D: usize>(&mut self, param: &Param<D>) {
            assert_eq!(param.id(), self.expected_id);
            assert_eq!(D, 2);
            assert!(!self.found, "visited same param twice");
            self.found = true;
        }
    }

    let mut visitor = MockParamVisitor {
        expected_id: module.param.id(),
        found: false,
    };
    module.visit_params(&mut visitor);

    assert!(visitor.found);
}
