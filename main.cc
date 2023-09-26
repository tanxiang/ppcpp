#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>

#include "loader.hh"
#include "log.hh"
#include "model.hh"

#include <tensorflow/cc/ops/array_ops.h>


int main(int argc, char* argv[])
{
    auto dataScope = tensorflow::Scope::NewRootScope().ExitOnError();
    auto cSession = tensorflow::ClientSession { dataScope };

    auto tfr = tfcc::getReader(dataScope,cSession,{"../data/test1.tfr"},64);


    std::vector<tensorflow::Tensor> lableImgs;

    auto runStatus = cSession.Run({}, { std::get<0>(tfr),  std::get<1>(tfr) }, &lableImgs);

    ALOG(MSG) <<runStatus.message() << '\t' << lableImgs[0].DebugString();

    runStatus = cSession.Run({}, { std::get<0>(tfr),  std::get<1>(tfr) }, &lableImgs);

    ALOG(MSG) <<runStatus.message() << '\t' << lableImgs[1].DebugString();

    //cSession.Run(tfr.node());

    auto nnScope = tensorflow::Scope::NewRootScope().ExitOnError();


    auto input0 = tensorflow::ops::Placeholder { nnScope, tensorflow::DT_FLOAT , tensorflow::ops::Placeholder::Shape({ 1, 128, 128, 3 }) };
    auto outputBox = tensorflow::ops::Placeholder { nnScope, tensorflow::DT_FLOAT , tensorflow::ops::Placeholder::Shape({ 1,  3200, 1 }) };

    auto output = tfcc::buildPPC(nnScope.NewSubScope("ppcpp"), input0.output);
    auto graph = dataScope.graph_as_shared_ptr();
    //auto cSession = tensorflow::ClientSession { rootScope };

    // tensorflow::ops::ApplyAdam adam{};

    return 0;
}
