cmake_minimum_required(VERSION 3.22)

project(ppcpp)

find_package(PkgConfig REQUIRED) 

pkg_check_modules(TFCC REQUIRED IMPORTED_TARGET tensorflow_cc)

add_executable(pcp main.cc)

set_property(TARGET pcp PROPERTY CXX_STANDARD 20)

target_include_directories(pcp PUBLIC ${TFCC_INCLUDE_DIRS})

target_link_libraries(pcp PUBLIC PkgConfig::TFCC)
