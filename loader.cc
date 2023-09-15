#include "loader.hh"
#include <algorithm>
#include <string>
#include <string_view>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/cc/ops/math_ops.h>

#include <tensorflow/cc/ops/data_flow_ops.h>
#include <tensorflow/cc/ops/experimental_dataset_ops.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/io_ops.h>
#include <tensorflow/cc/ops/parsing_ops.h>
#include <tensorflow/cc/training/queue_runner.h>

#include <tensorflow/core/example/example.pb.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.pb.h>

#include <tensorflow/core/platform/types.h>
#include <vector>

#include "log.hh"

namespace tfcc {

namespace tf = tensorflow;
namespace tfo = tensorflow::ops;


std::tuple<tensorflow::Output,tensorflow::Output> getReader(tf::Scope& scope,tf::ClientSession& cs,std::vector<std::string> fileNames,int64_t numRecoder)
{
    tfo::Placeholder fileNameSpace { scope.WithOpName("tfrFileName"), tf::DT_STRING };

    tfo::FIFOQueue queue { scope.WithOpName("inputQueue"), { tf::DT_STRING } };
    tfo::QueueEnqueue enQueue { scope.WithOpName("queueEn"), queue, { fileNameSpace.output } };
    tfo::QueueClose close { scope.WithOpName("queueClose"), queue };
    
    tfo::ReaderReadUpTo readm { scope.WithOpName("readQueueUpTo"), tfo::TFRecordReader { scope.WithOpName("tfrReader") }, queue, numRecoder };
    tfo::ParseExample examples {
        scope.WithOpName("examples"),
        { readm.values },tfo::Const<tf::string>(scope.WithOpName("noName"), {}, { 0 }),
        tf::gtl::ArraySlice<tf::Input> {}, { "image", "label" },
        {
            tfo::Const<tf::string>(scope.WithOpName("dense_def0"), {"",},{ 1 }),
            tfo::Const<tf::int64>(scope.WithOpName("dense_def1"), {1,},{ 1 }) 
        },
        {}, { { 1 }, { 1 } }};


    tfo::Unstack exampleImgRaw{scope.WithOpName("unstack"),examples.dense_values[0],64};
    std::vector<tf::Output> imageTensors;
    for(auto& imgRaw:exampleImgRaw.output){
        tfo::Cast imageTensorFloat{scope.WithOpName("floatCaster"), tfo::DecodeImage{scope,tfo::Unstack{scope.WithOpName("unstack"),imgRaw,1}[0]},tf::DT_FLOAT};
        tfo::ResizeBilinear imageTensorFloatResize{scope.WithOpName("resize"), tfo::ExpandDims{scope,imageTensorFloat,0}, tfo::Const(scope, { 128, 128 })};
        imageTensors.emplace_back( tfo::Div{scope.WithOpName("div"),imageTensorFloatResize,{255.f}});
    }
    tfo::Concat imageTensor{scope.WithOpName("concat"),tf::InputList{imageTensors},0};
    std::vector<tf::Tensor> tensorOut;
    for(auto&fileName:fileNames)
        auto runStatus = cs.Run({ { fileNameSpace, fileName } }, {}, { enQueue }, &tensorOut);
    auto runStatus = cs.Run({}, {}, { close }, &tensorOut);

    return {examples.dense_values[1],imageTensor};
}

}