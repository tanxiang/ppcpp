#pragma once
#include <cstdint>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/ops/parsing_ops.h>
#include <tuple>
#include <vector>
namespace tfcc {
std::tuple<tensorflow::Output, tensorflow::Output> getReader(tensorflow::Scope& scope, tensorflow::ClientSession& cs, std::vector<std::string>, int64_t);
}
