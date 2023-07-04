
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/gradients.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/ops/state_ops.h>
#include <tensorflow/core/platform/env.h>

#include <exception>
#include <iostream>
#include <memory>

namespace tfcc {

auto convBlock(tensorflow::Scope &scope, tensorflow::Tensor inputs, int filters,
               float alpha, std::array<int, 2> kernelShape = { 3, 3 },
               std::array<int, 2> strides = { 1, 1 }) {
  auto convPad =
      tensorflow::ops::Pad(scope.WithOpName("pad0"), inputs,
                           { { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } });

  auto weightsInitial = tensorflow::ops::RandomNormal(
      scope, tensorflow::ops::Const(scope,{kernelShape[0],kernelShape[1]} ), inputs.dtype());

  auto weight = tensorflow::ops::Variable(
      scope, {filters, kernelShape[0], kernelShape[1], 1 }, inputs.dtype());

  tensorflow::ops::Assign(scope, weight, weightsInitial);
  auto biases = tensorflow::ops::Variable(scope, { filters }, inputs.dtype());
  tensorflow::Output o;
}

auto convBlock(tensorflow::Scope &scope, tensorflow::Input& inputs, int filters,
               float alpha, std::array<int, 2> kernelShape = { 3, 3 },
               std::array<int, 2> strides = { 1, 1 }) {
  auto convPad =
      tensorflow::ops::Pad(scope.WithOpName("pad0"), inputs,
                           { { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } });

  auto weightsInitial = tensorflow::ops::RandomNormal(
      scope, tensorflow::ops::Const(scope,{kernelShape[0],kernelShape[1]} ), inputs.data_type());

  auto weight = tensorflow::ops::Variable(
      scope, { filters, kernelShape[0], kernelShape[1], 1 }, inputs.data_type());

  tensorflow::ops::Assign(scope, weight, weightsInitial);
  auto biases = tensorflow::ops::Variable(scope, { filters }, inputs.data_type());
  tensorflow::Output o;
}


auto buildInputBlocks(tensorflow::Scope scope,
                      tensorflow::ops::Placeholder &input) {
  return tensorflow::Tensor{};
}

auto buildPPC(const tensorflow::Scope &rootScope,
              tensorflow::ops::Placeholder &input,
              tensorflow::ops::Placeholder &output) {
  auto input1 = buildInputBlocks(rootScope.NewSubScope("input0"), input);
}

}  // namespace tfcc

int main(int argc, char *argv[]) {
  auto rootScope = tensorflow::Scope::NewRootScope();

  auto input0 = tensorflow::ops::Placeholder{
    rootScope, tensorflow::DT_INT16,
    tensorflow::ops::Placeholder::Shape({ 1, 512, 512, 1 })
  };

  auto outputBox = tensorflow::ops::Placeholder{
    rootScope, tensorflow::DT_INT16,
    tensorflow::ops::Placeholder::Shape({ 1, 4, 3200, 1 })
  };

  tfcc::buildPPC(rootScope, input0, outputBox);

  auto graph = rootScope.graph_as_shared_ptr();

  auto cSession = tensorflow::ClientSession{ rootScope };

  return 0;
}
