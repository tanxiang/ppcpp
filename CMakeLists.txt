cmake_minimum_required(VERSION 3.22)

project(ppcpp)

find_package(PkgConfig REQUIRED) 

pkg_check_modules(TFCC REQUIRED IMPORTED_TARGET tensorflow_cc)

add_executable(pcp main.cc loader.cc model.cc)

add_executable(test test.cc)


set_property(TARGET pcp PROPERTY CXX_STANDARD 20)

set_property(TARGET test PROPERTY CXX_STANDARD 20)


target_include_directories(pcp PUBLIC ${TFCC_INCLUDE_DIRS})

target_include_directories(test PUBLIC ${TFCC_INCLUDE_DIRS})


target_link_libraries(pcp PUBLIC PkgConfig::TFCC)

target_link_libraries(test PUBLIC PkgConfig::TFCC)
