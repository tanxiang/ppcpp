#pragma once
#include <tensorflow/cc/ops/io_ops.h>
namespace tfcc {
tensorflow::ops::ReaderRead getReader(tensorflow::Scope& scope);
}
