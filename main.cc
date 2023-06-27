
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

#include <tensorflow/cc/framework/gradients.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/client/client_session.h>

#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/state_ops.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/nn_ops.h>

#include <iostream>
#include <memory>
#include <exception>

namespace tfcc {
std::unique_ptr<tensorflow::Session> Session() {
  tensorflow::Session *session = nullptr;
  auto status = tensorflow::NewSession({}, &session);
  if (!status.ok()) {
    throw std::logic_error{status.ToString()};
  } 
  return std::unique_ptr<tensorflow::Session>{session};

}
}  // namespace tfcc

int main(int argc, char *argv[]) {
  auto session= tfcc::Session();
  return 0;
}
