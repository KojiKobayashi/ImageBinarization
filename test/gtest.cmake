cmake_minimum_required(VERSION 2.8)

# Google Test settings
include(ExternalProject)

ExternalProject_Add(
    GoogleTest
    URL https://github.com/google/googletest/archive/release-1.8.0.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/lib
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
    )

ExternalProject_Get_Property(GoogleTest source_dir)
include_directories(${source_dir}/googletest/include)
include_directories(${source_dir}/googlemock/include)

ExternalProject_Get_Property(GoogleTest binary_dir)
add_library(gtest STATIC IMPORTED)
set_property(
    TARGET gtest
    PROPERTY IMPORTED_LOCATION ${binary_dir}/googlemock/gtest/Release/gtest.lib
    )
add_library(gtest_main STATIC IMPORTED)
set_property(
    TARGET gtest_main
    PROPERTY IMPORTED_LOCATION ${binary_dir}/googlemock/gtest/Release/gtest_main.lib
    )
    #[[
add_library(gmock STATIC IMPORTED)
set_property(
    TARGET gmock
    PROPERTY IMPORTED_LOCATION ${binary_dir}/googlemock/libgmock.a
    )
add_library(gmock_main STATIC IMPORTED)
set_property(
    TARGET gmock_main
    PROPERTY IMPORTED_LOCATION ${binary_dir}/googlemock/libgmock_main.a
    )
    ]]