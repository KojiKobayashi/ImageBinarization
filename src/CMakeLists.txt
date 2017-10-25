cmake_minimum_required(VERSION 2.8)

find_package(OpenCV 3.2.0 REQUIRED)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release")

add_library(ImageBinarization STATIC
	NatBinarization.cpp
	NiblackBinarization.cpp
)
