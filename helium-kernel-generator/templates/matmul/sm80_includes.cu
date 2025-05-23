#include <cute/tensor_impl.hpp>
#include <cuda_runtime.h>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cute/arch/mma.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cute/tensor.hpp>

using namespace cute;

// template <typename T> struct always_false : std::false_type {};
//
// template <typename T> void debug_type(T &&var) {
//   static_assert(always_false<T>::value, "debug type information:");
// }