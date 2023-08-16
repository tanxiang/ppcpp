#include <exception>
#include <iostream>
#include <memory>
#include <string_view>

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/gradients.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/ops/state_ops.h>
#include <tensorflow/core/platform/env.h>

namespace tfcc {

namespace tf = tensorflow;
namespace tfo = tensorflow::ops;

auto imgDecode(std::string_view fileName, std::string_view inputName)
{
    auto root = tf::Scope::NewRootScope();
    auto file_reader = tfo::ReadFile(root.WithOpName(inputName), std::string { fileName });
    tf::Output image_reader;

    if (fileName.ends_with(".png")) {
        image_reader = tfo::DecodePng(root.WithOpName("pngReader"), file_reader, tfo::DecodePng::Channels(1));
    } else if (fileName.ends_with(".gif")) {
        image_reader = tfo::DecodeGif(root.WithOpName("gifReader"), file_reader);
    } else {
        image_reader = tfo::DecodeJpeg(root.WithOpName("jpgReader"), file_reader);
    }
    return image_reader;
}

auto active(tf::Scope& scope, tf::Output inputs)
{
    std::cout<<inputs.name()<<'\n';

    auto relu = tfo::Relu6(scope.WithOpName("relu6"), inputs);

    return relu;
}

auto Dense(tf::Scope scope, tf::Output inputs, int out_units)
{
    auto inputShape = tfo::Shape(scope.WithOpName("shape"), inputs);
    auto weightShape = tfo::Concat(scope.WithOpName("apppend"), { inputShape.output, tfo::Const(scope, { out_units }) }, 0);
    auto weightsInitial = tfo::RandomNormal(scope, weightShape, inputShape.output.type());

    auto weight = tfo::Variable(scope.WithOpName("weight"), {}, inputs.type());
    
    tfo::Assign(scope, weight, weightsInitial);

    auto biases = tfo::Variable(scope.WithOpName("biases"), { out_units }, inputs.type());
    tfo::Assign(scope, biases, tf::Input::Initializer(0.f, { out_units }));

    return tfo::Add(scope.WithOpName("AddBiases"), tfo::MatMul(scope.WithOpName("MatMulWeight"), inputs, weight), biases);
}

auto Conv(tf::Scope scope, tf::Output inputs, int filters, int channel = 1, std::array<int, 2> kernelShape = { 3, 3 }, std::array<int, 2> strides = { 1, 1 })
{
    //inputs = tfo::Pad(scope.WithOpName("pad0"), inputs, { { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } });

    auto inputShape = tfo::Shape(scope.WithOpName("Shape"), inputs);
    std::cout << inputShape.node()->DebugString() << std::endl;

    auto weightsInitial = tfo::RandomNormal(scope, tfo::Const(scope, { filters, kernelShape[0], kernelShape[1], channel  }), inputs.type());

    auto weight = tfo::Variable(scope, { filters, kernelShape[0], kernelShape[1], channel }, inputs.type());
    tfo::Assign(scope, weight, weightsInitial);


    auto convOutput = tfo::Conv2D(scope.WithOpName("Conv"), inputs, weight, { filters, strides[0], strides[1], channel }, std::string { "SAME" });
    //if(convOutput.node())        std::cout << convOutput.node()->DebugString() << std::endl;
    std::cout << convOutput.node()->DebugString() << std::endl;

    auto convShape = tfo::Shape(scope.WithOpName("Shape"), convOutput.output);
    std::cout << convShape.operation.input_type(0)<< std::endl;

    auto biases = tfo::Variable(scope, { filters }, inputs.type());
    tfo::Assign(scope, biases, tf::Input::Initializer(0.f, { filters }));

    //return convOutput;
    auto badd = tfo::BiasAdd(scope.WithOpName("bias"), convOutput, biases);
    if(badd.node()){

    }
    return active(scope, badd);
}

auto Dropout(tf::Scope scope, tf::Input inputs) { }

auto Flatten(tf::Scope scope, tf::Input inputs)
{
    return tfo::Reshape(scope.WithOpName("reshape"), inputs, { -1 });
}

auto Gap(tf::Scope scope, tf::Input inputs)
{
    auto inputShape = tfo::Shape(scope.WithOpName("shape"), inputs);
    // return tfo::AvgPool(scope, inputs, {-1});
}

auto buildInputBlocks(tf::Scope scope,tf::Output input)
{
    return tfcc::Conv(scope.NewSubScope("conv0"), input, 12);
}

auto buildPPC(tf::Scope rootScope, tfo::Placeholder& input,
    tfo::Placeholder& output)
{
    tf::Output x = buildInputBlocks(rootScope.NewSubScope("head"), input);
    //std::cout<<x.operator tensorflow::Output().type();
    
   // x = buildInputBlocks(rootScope.NewSubScope("head1"), x);
  //  x = buildInputBlocks(rootScope.NewSubScope("head2"), x);
   //x = Flatten(rootScope.NewSubScope("head3"),x);
    //x = Dense(rootScope.NewSubScope("body0"),x,256);

    return x;
}

} // namespace tfcc

int main(int argc, char* argv[])
{
    auto rootScope = tensorflow::Scope::NewRootScope().ExitOnError();
    auto input0 = tensorflow::ops::Placeholder { rootScope, tensorflow::DT_FLOAT, tensorflow::ops::Placeholder::Shape({ 1, 512, 512, 1 }) };
    auto outputBox = tensorflow::ops::Placeholder { rootScope, tensorflow::DT_FLOAT, tensorflow::ops::Placeholder::Shape({ 1, 4, 3200, 1 }) };

    auto output = tfcc::buildPPC(rootScope.NewSubScope("ppcpp"), input0, outputBox);
    auto graph = rootScope.graph_as_shared_ptr();
    auto cSession = tensorflow::ClientSession { rootScope };

    //tensorflow::ops::ApplyAdam adam{};

    return 0;
}
