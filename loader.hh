#pragma once
#include <tensorflow/cc/ops/dataset_ops_internal.h>
namespace tfcc {
tensorflow::ops::internal::TFRecordDataset getReader(tensorflow::Scope& scope);
}
