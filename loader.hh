#pragma once
#include <tensorflow/cc/ops/parsing_ops.h>
namespace tfcc {
tensorflow::ops::ParseSingleExample getReader(tensorflow::Scope& scope);
}
