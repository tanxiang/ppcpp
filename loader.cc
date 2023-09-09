#include "loader.hh"
#include <string>
#include <string_view>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/const_op.h>
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

tfo::ParseExample getReader(tf::Scope& scope,const std::string fileName,int64_t numRecoder)
{
        tf::ClientSession cs { scope };

    // tfo::TFRecordReader reader { scope ,tfo::TFRecordReader::Attrs{}.SharedName("trainData")};

    tfo::Placeholder fileNameSpace { scope.WithOpName("tfrFileName"), tf::DT_STRING };

    tfo::FIFOQueue queue { scope.WithOpName("inputQueue"), { tf::DT_STRING } };

    tfo::QueueEnqueue en { scope.WithOpName("queueEn"), queue, { fileNameSpace.output } };

    tfo::QueueClose close { scope.WithOpName("queueClose"), queue };

    tfo::QueueSize queueSize { scope.WithOpName("sizeofQueue"), queue };

    /*tfo::ReaderRead read { scope.WithOpName("readQueue"), tfo::TFRecordReader { scope.WithOpName("tfrReader") }, queue };

    tfo::ParseSingleExample example { scope.WithOpName("example"),
        { read.value },
        { tfo::Const<tf::string>(scope.WithOpName("dense_def0"), "", { 1 }),
            tfo::Const<tf::int64>(scope.WithOpName("dense_def1"), 1, { 1 }) },
        0, {}, { "image", "label" }, {}, { { 1 }, { 1 } } };
*/
    // tfo::ParseExample examples{scope.WithOpName("examples"),{readm.values},{},};
    tfo::ReaderReadUpTo readm { scope.WithOpName("readQueueUpTo"), tfo::TFRecordReader { scope.WithOpName("tfrReader") }, queue, numRecoder };

    tfo::ParseExample examples {
        scope.WithOpName("examples"),
        { readm.values },tfo::Const<tf::string>(scope.WithOpName("noName"), {}, { 0 }),
        tf::gtl::ArraySlice<tf::Input> {}, { "image", "label" },
        {
            tfo::Const<tf::string>(scope.WithOpName("dense_def0"), {"",},{ 1 }),
            tfo::Const<tf::int64>(scope.WithOpName("dense_def1"), {1,},{ 1 }) 
        },
        {}, { { 1 }, { 1 } }
         };

    std::vector<tf::Tensor> tensorOut;
    auto runStatus = cs.Run({ { fileNameSpace, fileName } }, {}, { en }, &tensorOut);
    runStatus = cs.Run({}, {}, { close }, &tensorOut);
    runStatus = cs.Run({}, { queueSize }, &tensorOut);

    ALOG(MSG) << tensorOut.size() << '\t' << tensorOut[0].DebugString();

    std::vector<tf::Tensor> imgs;


    runStatus = cs.Run({}, { examples.dense_values[0], examples.dense_values[1] }, &imgs);

    ALOG(MSG) << imgs.size() << '\t' << imgs[1].DebugString();


    //    tf::QueueRunner::New(queue_runner_def, &qr);
    //    tf::ClientSession cs;
    return examples;
}

}