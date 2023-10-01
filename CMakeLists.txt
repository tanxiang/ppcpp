cmake_minimum_required(VERSION 3.22)


set(CMAKE_CXX_STANDARD 20)# turn on the dynamic depends for ninja
set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
# turn on the experimental API
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API  "2182bf5c-ef0d-489a-91da-49dbc3090d2a")
#set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FORMAT "gcc")
#set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FLAG  "${compiler_flags_for_module_map} -fmodule-mapper=<MODULE_MAP_FILE>")

project(ppcpp)

find_package(PkgConfig REQUIRED) 

pkg_check_modules(TFCC REQUIRED IMPORTED_TARGET tensorflow_cc)

add_executable(pcp main.cc loader.cc model.cc)

#target_compile_options(pcp PUBLIC -fmodules-ts)

add_executable(test test.cc)


set_property(TARGET pcp PROPERTY CXX_STANDARD 23)

set_property(TARGET test PROPERTY CXX_STANDARD 23)


target_include_directories(pcp PUBLIC ${TFCC_INCLUDE_DIRS})

target_include_directories(test PUBLIC ${TFCC_INCLUDE_DIRS})


target_link_libraries(pcp PUBLIC PkgConfig::TFCC)

target_link_libraries(test PUBLIC PkgConfig::TFCC)
