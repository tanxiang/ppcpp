#include <tensorflow/cc/ops/standard_ops.h>

using namespace tensorflow;
using namespace tensorflow::ops;

#ifdef ABSL_HAVE_STD_STRING_VIEW 
#error "ABSL_HAVE_STD_STRING_VIEW is set." 
#endif
int main() {
  auto scope = Scope::NewRootScope();
  auto input = Const(scope, Tensor(DataType::DT_INT32, {}));
  auto filter = Const(scope, Tensor(DataType::DT_INT32, {}));
  Conv2D(scope, input, filter, { 1, 1, 1, 1 }, "SAME");
}