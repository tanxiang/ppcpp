#include "loader.hh"
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/ops/dataset_ops.h>
#include <tensorflow/cc/ops/experimental_dataset_ops.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/io_ops.h>

namespace tfcc {

namespace tf = tensorflow;
namespace tfo = tensorflow::ops;

auto getReader(tf::Scope& scope)
{
    tfo::TFRecordReader reader { scope };
}

}