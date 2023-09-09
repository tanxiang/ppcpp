#pragma once
#include <cstdint>
#include <tensorflow/cc/ops/parsing_ops.h>
namespace tfcc {
tensorflow::ops::ParseExample getReader(tensorflow::Scope& scope,std::string ,int64_t );
}
