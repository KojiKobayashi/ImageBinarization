cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_LESS "3.0.0")
  message(FATAL_ERROR "ERROR: Can't find OpenCV 3.0.0+")
endif()

set(CMAKE_CONFIGURATION_TYPES "Debug;Release")

add_library(ImageBinarization STATIC
	NatBinarization.cpp
	NiblackBinarization.cpp
	NickBinarization.cpp
	SauvolaBinarization.cpp
	NatBinarization.h
	NiblackBinarization.h
	NickBinarization.h
	SauvolaBinarization.h
)
