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
#include <tensorflow/core/framework/attr_value_util.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/graph/graph.h>
#include <tensorflow/core/platform/env.h>

namespace tfcc {

namespace tf = tensorflow;
namespace tfo = tensorflow::ops;

struct blockOutput{
    tf::Output output;
    int32_t channelNum;
};

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

blockOutput Dense(tf::Scope scope, tf::Output inputs, int inChannel , int outChannel)
{
    auto weightsInitial = tfo::RandomNormal(scope, {inChannel,outChannel}, inputs.type());

    auto weight = tfo::Variable(scope.WithOpName("weight"), {inChannel,outChannel}, inputs.type());

    tfo::Assign(scope.WithOpName("assignWeight"), weight, weightsInitial);

    auto biases = tfo::Variable(scope.WithOpName("biases"), { outChannel }, inputs.type());
    tfo::Assign(scope, biases, tf::Input::Initializer(0.f, { outChannel }));

    return {tfo::Add{scope.WithOpName("AddBiases"), tfo::MatMul(scope.WithOpName("MatMulWeight"), inputs, weight), biases},outChannel};
}

blockOutput Conv(tf::Scope scope, tf::Output& inputs,int inChannel , int outChannel,  std::array<int, 2> kernelShape = { 3, 3 }, std::array<int, 4> strides = { 1,1,1, 1 })
{
    // inputs = tfo::Pad(scope.WithOpName("pad0"), inputs, { { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } });

    tfo::Stack kShape{scope, { kernelShape[0], kernelShape[1], inChannel, outChannel }};
    auto weightsInitial = tfo::RandomNormal(scope, kShape, inputs.type());
    auto weight = tfo::Variable(scope, { kernelShape[0], kernelShape[1], inChannel, outChannel }, inputs.type());
    tfo::Assign(scope, weight, weightsInitial);

    tfo::Conv2D convOutput = tfo::Conv2D(scope.WithOpName("Conv"), inputs, weight, strides, std::string { "SAME" });
    // if(convOutt.node())        std::cout << convOutput.node()->DebugString() << std::endl;
    ALOG(MSG) << convOutput.node()->DebugString();

    auto convShape = tfo::Shape(scope.WithOpName("Shape"), convOutput.output);
    ALOG(MSG) << convShape.output.node()->DebugString();

    auto biases = tfo::Variable(scope, { outChannel }, inputs.type());
    tfo::Assign(scope, biases, tf::Input::Initializer(0.f, { outChannel }));

    // return convOutput;
    auto badd = tfo::BiasAdd(scope.WithOpName("bias"), convOutput.output, biases);

    return {tfo::MaxPool{scope.WithOpName("Pool"),active(scope, badd),{ 1, 2, 2, 1 }, { 1, 2, 2, 1 }, "SAME"},outChannel};
}

auto Dropout(tf::Scope scope, tf::Input inputs) { }

blockOutput Flatten(tf::Scope scope, tf::Input inputs,int inChannel , int outChannel)
{
    
    return {tfo::Reshape{scope.WithOpName("reshape"), inputs, { -1, outChannel}},outChannel};
}

auto Gap(tf::Scope scope, tf::Input inputs)
{
    auto inputShape = tfo::Shape(scope.WithOpName("shape"), inputs);
    // return tfo::AvgPool(scope, inputs, {-1});
}

auto buildInputBlocks(tf::Scope scope, tensorflow::Output& input,int32_t iChannelNum,int32_t oChannelNum)
{
    return tfcc::Conv(scope.NewSubScope("conv0"), input, iChannelNum,oChannelNum);
}

tf::Output buildPPC(tf::Scope scope, tensorflow::Output& input)
{
    auto inputShape = input.node()->attrs().Find("shape")->shape();
    ALOG(MSG)<<inputShape.dim(3).size();
    auto x = buildInputBlocks(scope.NewSubScope("head"), input,inputShape.dim(3).size(),12);
    tfo::Placeholder labels { scope.NewSubScope("head"), tf::DataType::DT_INT32 };

    x = buildInputBlocks(scope.NewSubScope("head1"), x.output,x.channelNum,128);
    x = buildInputBlocks(scope.NewSubScope("head2"), x.output,x.channelNum,512);
    x = Flatten(scope.NewSubScope("head3"), x.output,x.channelNum,1024);
    x = Dense(scope.NewSubScope("body0"), x.output,x.channelNum, 3560);
    x = Dense(scope.NewSubScope("body1"), x.output,x.channelNum, 3700);

    tfo::SparseSoftmaxCrossEntropyWithLogits CSWL { scope.WithOpName("crossEntropy"), x.output, labels };

    for (const auto& node : scope.graph()->nodes()) {
        if (node->type_string() == "VariableV2") {
            ALOG(ERROR) << node->DebugString();
            auto attrs = node->attrs();
            attrs.Find("dtype");
            if (auto dType = attrs.Find("dtype"))
                if (auto dShape = attrs.Find("shape")) {
                    ALOG(MSG) << dType->type()<< " "<<dShape->shape();
                }
        }
    }
    auto status = tf::AddSymbolicGradients(scope.NewSubScope("gradient"), {}, {}, nullptr);

    return x.output;
}

} // namespace tfcc
