#pragma once
#include <tensorflow/cc/framework/scope.h>



namespace tfcc {
tensorflow::Output buildPPC(tensorflow::Scope rootScope, tensorflow::Output& input);
}