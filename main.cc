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

auto imgDecode(std::string_view fileName, std::string_view inputName) {
  auto root = tensorflow::Scope::NewRootScope();
  auto file_reader = tensorflow::ops::ReadFile(root.WithOpName(inputName),
                                               std::string{ fileName });
  tensorflow::Output image_reader;

  if (tensorflow::StringPiece(fileName).ends_with(".png")) {
    image_reader =
        tensorflow::ops::DecodePng(root.WithOpName("pngReader"), file_reader,
                                   tensorflow::ops::DecodePng::Channels(1));
  } else if (tensorflow::StringPiece(fileName).ends_with(".gif")) {
    image_reader =
        tensorflow::ops::DecodeGif(root.WithOpName("gifReader"), file_reader);
  } else {
    image_reader =
        tensorflow::ops::DecodeJpeg(root.WithOpName("jpgReader"), file_reader);
  }
  return image_reader;
}

auto active(tensorflow::Scope &scope, tensorflow::Input inputs) {
  return tensorflow::ops::Relu6(scope.WithOpName("relu6"), inputs);
}

auto dense(tensorflow::Scope &scope, tensorflow::Input inputs, int in_units,
           int out_units) {
  auto weightsInitial = tensorflow::ops::RandomNormal(
      scope, tensorflow::ops::Const(scope, { in_units, out_units }),
      inputs.data_type());
  inputs.tensor().dim_size(0);
  auto weight = tensorflow::ops::Variable(
      scope.WithOpName("weight"), { in_units, out_units }, inputs.data_type());

  tensorflow::ops::Assign(scope, weight, weightsInitial);

  auto biases = tensorflow::ops::Variable(scope.WithOpName("biases"),
                                          { out_units }, inputs.data_type());
  tensorflow::ops::Assign(scope, biases,
                          tensorflow::Input::Initializer(0.f, { out_units }));
  return tensorflow::ops::Add(
      scope.WithOpName("AddBiases"),
      tensorflow::ops::MatMul(scope.WithOpName("MatMulWeight"), inputs, weight),
      biases);
}

auto convPack(tensorflow::Scope scope, tensorflow::Output inputs, int filters,
              std::array<int, 2> kernelShape = { 3, 3 },
              std::array<int, 2> strides = { 1, 1 }) {
  auto convPad =
      tensorflow::ops::Pad(scope.WithOpName("pad0"), inputs,
                           { { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } });
  std::cout << inputs.type() << std::endl;

  auto weightsInitial = tensorflow::ops::RandomNormal(
      scope, tensorflow::ops::Const(scope, { kernelShape[0], kernelShape[1] }),
      inputs.type());

  auto weight = tensorflow::ops::Variable(
      scope, { filters, kernelShape[0], kernelShape[1], 1 },
      inputs.type());
  tensorflow::ops::Assign(scope, weight, weightsInitial);

  auto biases =
      tensorflow::ops::Variable(scope, { filters }, inputs.type());
  tensorflow::ops::Assign(scope, biases,
                          tensorflow::Input::Initializer(0.f, { filters }));

  auto convOutput = tensorflow::ops::Conv2D(
      scope.WithOpName("conv"), convPad, weight,
      { 1, strides[0], strides[1], 1 }, std::string{ "SAME" });

  return active(scope, convOutput);
}

auto Dropout(tensorflow::Scope scope, tensorflow::Input inputs) {}

auto Flatten(tensorflow::Scope scope, tensorflow::Input inputs) {
  auto flat =
      tensorflow::ops::Reshape(scope.WithOpName("reshape"), inputs, { -1, 99 });
}

auto Gap(tensorflow::Scope scope, tensorflow::Input inputs) {
  // auto flat = tensorflow::ops::AvgPool(scope, inputs, {-1, 88});
}

auto buildInputBlocks(tensorflow::Scope scope,
                      tensorflow::ops::Placeholder &input) {

  return tfcc::convPack(scope.NewSubScope("conv0"), input, 1);
}

auto buildPPC(tensorflow::Scope rootScope, tensorflow::ops::Placeholder &input,
              tensorflow::ops::Placeholder &output) {
  auto output0 = buildInputBlocks(rootScope.NewSubScope("head"), input);
  return output0;
}

}  // namespace tfcc

int main(int argc, char *argv[]) {
  auto rootScope = tensorflow::Scope::NewRootScope();

  auto input0 = tensorflow::ops::Placeholder{
    rootScope, tensorflow::DT_FLOAT,
    tensorflow::ops::Placeholder::Shape({ 1, 512, 512, 1 })
  };

  auto outputBox = tensorflow::ops::Placeholder{
    rootScope, tensorflow::DT_FLOAT,
    tensorflow::ops::Placeholder::Shape({ 1, 4, 3200, 1 })
  };

  auto output =
      tfcc::buildPPC(rootScope.NewSubScope("ppcpp"), input0, outputBox);

  auto graph = rootScope.graph_as_shared_ptr();

  auto cSession = tensorflow::ClientSession{ rootScope };

  // tensorflow::ops::ApplyAdam adam{};

  return 0;
}
