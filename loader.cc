#include "loader.hh"
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/cc/ops/experimental_dataset_ops.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/io_ops.h>
#include <tensorflow/cc/ops/array_ops.h>

#include "log.hh"

namespace tfcc {

namespace tf = tensorflow;
namespace tfo = tensorflow::ops;

tfo::internal::TFRecordDataset getReader(tf::Scope& scope)
{
    auto fileNameVar = tfo::Placeholder(scope.WithOpName("input"), tf::DT_STRING);

    tfo::TFRecordReader reader { scope ,tfo::TFRecordReader::Attrs{}.SharedName("trainData")};

    tfo::ReaderRead read{scope, tfo::TFRecordReader{scope},tfo::Const(scope,{"../../tfr"})};

    tfo::internal::TFRecordDataset dataset{scope,fileNameVar,tfo::Const(scope,""),tfo::Const(scope,0ll)};
    ALOG(MSG)<<dataset.node()->DebugString();
    return dataset;
}

}