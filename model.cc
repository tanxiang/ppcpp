#include "model.hh"
#include "log.hh"
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/gradients.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/nn_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/ops/state_ops.h>
#include <tensorflow/core/graph/graph.h>
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
    // std::cout << inputs.name() << '\n';

    auto relu = tfo::Relu6(scope.WithOpName("relu6"), inputs);

    return relu;
}

auto Dense(tf::Scope scope, tf::Output inputs, int out_units)
{
    auto inputShape = tfo::Shape(scope.WithOpName("shape"), inputs);
    tfo::Split inputShapes{scope.WithOpName("shapeSplit"),0,inputShape,2};
    auto weightShape = tfo::Concat(scope.WithOpName("apppend"), { inputShapes[0], tfo::Const(scope, { out_units }) }, 0);
    auto weightsInitial = tfo::RandomNormal(scope, weightShape, inputs.type());

    auto weight = tfo::Variable(scope.WithOpName("weight"), {-1,out_units}, inputs.type());

    tfo::Assign(scope.WithOpName("assignWeight"), weight, weightsInitial);

    auto biases = tfo::Variable(scope.WithOpName("biases"), { out_units }, inputs.type());
    tfo::Assign(scope, biases, tf::Input::Initializer(0.f, { out_units }));

    return tfo::Add(scope.WithOpName("AddBiases"), tfo::MatMul(scope.WithOpName("MatMulWeight"), inputs, weight), biases);
}

auto Conv(tf::Scope scope, tf::Output& inputs, int filters, int inChannel = 1, std::array<int, 2> kernelShape = { 3, 3 }, std::array<int, 2> strides = { 1, 1 })
{
    // inputs = tfo::Pad(scope.WithOpName("pad0"), inputs, { { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } });

    // auto inputShape = tfo::Shape(scope.WithOpName("Shape"), inputs);
    ALOG(MSG) << inputs.node()->DebugString();
    auto weightsInitial = tfo::RandomNormal(scope, tfo::Const(scope, { kernelShape[0], kernelShape[1], inChannel, filters }), inputs.type());
    auto weight = tfo::Variable(scope, { kernelShape[0], kernelShape[1], inChannel, filters }, inputs.type());
    tfo::Assign(scope, weight, weightsInitial);

    tfo::Conv2D convOutput = tfo::Conv2D(scope.WithOpName("Conv"), inputs, weight, { filters, strides[0], strides[1], inChannel }, std::string { "SAME" });
    // if(convOutt.node())        std::cout << convOutput.node()->DebugString() << std::endl;
    ALOG(MSG) << convOutput.node()->DebugString();

    auto convShape = tfo::Shape(scope.WithOpName("Shape"), convOutput.output);
    ALOG(MSG) << convShape.output.node()->DebugString();

    auto biases = tfo::Variable(scope, { filters }, inputs.type());
    tfo::Assign(scope, biases, tf::Input::Initializer(0.f, { filters }));

    // return convOutput;
    auto badd = tfo::BiasAdd(scope.WithOpName("bias"), convOutput.output, biases);
    if (badd.node()) {
    }
    return active(scope, badd);
}

auto Dropout(tf::Scope scope, tf::Input inputs) { }

auto Flatten(tf::Scope scope, tf::Input inputs)
{
    return tfo::Reshape(scope.WithOpName("reshape"), inputs, { -1 ,1});
}

auto Gap(tf::Scope scope, tf::Input inputs)
{
    auto inputShape = tfo::Shape(scope.WithOpName("shape"), inputs);
    // return tfo::AvgPool(scope, inputs, {-1});
}

auto buildInputBlocks(tf::Scope scope, tensorflow::Output& input)
{
    return tfcc::Conv(scope.NewSubScope("conv0"), input, 12);
}

tf::Output buildPPC(tf::Scope rootScope, tensorflow::Output& input)
{
    tf::Output x = buildInputBlocks(rootScope.NewSubScope("head"), input);

    x = buildInputBlocks(rootScope.NewSubScope("head1"), x);
    x = buildInputBlocks(rootScope.NewSubScope("head2"), x);
    x = Flatten(rootScope.NewSubScope("head3"),x);
    x = Dense(rootScope.NewSubScope("body0"),x,256);

    return x;
}

} // namespace tfcc
