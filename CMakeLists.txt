cmake_minimum_required(VERSION 2.8)

project(ImageBinarization)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MT")

add_subdirectory(src)
add_subdirectory(test)
enable_testing()
add_test(NAME ImageBinarizationTest COMMAND Test)
