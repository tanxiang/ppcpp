#pragma once
#include <cstdint>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/ops/parsing_ops.h>
#include <tuple>
namespace tfcc {
//tensorflow::ops::ParseExample getReader(tensorflow::Scope& scope,std::string ,int64_t );

std::tuple<tensorflow::Output,tensorflow::Output> getReader(tensorflow::Scope& scope,tensorflow::ClientSession& cs,std::string ,int64_t );

}
