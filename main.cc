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

    auto imgDecode(std::string_view fileName,std::string_view inputName){
	auto root = tensorflow::Scope::NewRootScope();
	auto file_reader = tensorflow::ops::ReadFile(root.WithOpName(inputName), std::string{fileName});
       	tensorflow::Output image_reader;

	if (tensorflow::StringPiece(fileName).ends_with(".png")) {
		image_reader = tensorflow::ops::DecodePng(root.WithOpName("png_reader"), file_reader,
								tensorflow::ops::DecodePng::Channels(1));
    }else if(tensorflow::StringPiece(fileName).ends_with(".gif")){
		image_reader = tensorflow::ops::DecodeGif(root.WithOpName("gif_reader"), file_reader);
    }else{
		image_reader = tensorflow::ops::DecodeJpeg(root.WithOpName("jpg_reader"), file_reader);
    }
        return image_reader;
    }

auto active(tensorflow::Scope &scope, tensorflow::Input inputs) {
  return tensorflow::ops::Relu6(scope, inputs);
}

auto denseBlock(tensorflow::Scope &scope, tensorflow::Input inputs, int in_units, int out_units) {
  auto weight =  tensorflow::ops::Variable(scope.WithOpName("weight"), {in_units, out_units}, tensorflow::DT_FLOAT);
  auto ba = tensorflow::ops::Variable(scope.WithOpName("Ba"), { out_units}, tensorflow::DT_FLOAT);
  return tensorflow::ops::Add( scope.WithOpName("AddB"),tensorflow::ops::MatMul(scope.WithOpName("MatMulW"), inputs,weight),ba);

}

auto convBlock(tensorflow::Scope &scope, tensorflow::Input inputs, int filters,
               float alpha, std::array<int, 2> kernelShape = { 3, 3 },
               std::array<int, 2> strides = { 1, 1 }) {
  auto convPad =
      tensorflow::ops::Pad(scope.WithOpName("pad0"), inputs,
                           { { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } });

  auto weightsInitial = tensorflow::ops::RandomNormal(
      scope, tensorflow::ops::Const(scope, { kernelShape[0], kernelShape[1] }),
      inputs.data_type());

  auto weight = tensorflow::ops::Variable(
      scope, { filters, kernelShape[0], kernelShape[1], 1 },
      inputs.data_type());

  tensorflow::ops::Assign(scope, weight, weightsInitial);
  auto biases =
      tensorflow::ops::Variable(scope, { filters }, inputs.data_type());
  auto convOutput =
      tensorflow::ops::Conv2D(scope.WithOpName("conv1"), convPad, weight,
                              { 1, strides[0], strides[1], 1 },std::string{ "SAME"});

  return active(scope, convOutput);
}

auto buildInputBlocks(tensorflow::Scope scope,
                      tensorflow::ops::Placeholder &input) {
  return tensorflow::Tensor{};
}

auto buildPPC(const tensorflow::Scope &rootScope,
              tensorflow::ops::Placeholder &input,
              tensorflow::ops::Placeholder &output) {
  auto output0 = buildInputBlocks(rootScope.NewSubScope("input0"), input);
  return output0;
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

  auto output =  tfcc::buildPPC(rootScope, input0, outputBox);

  auto graph = rootScope.graph_as_shared_ptr();

  auto cSession = tensorflow::ClientSession{ rootScope };

  //tensorflow::ops::ApplyAdam adam{};

  return 0;
}
