cmake_minimum_required(VERSION 3.23)
project(untitled)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -fopenmp)

add_executable(untitled main.cpp DegreesOfFreedom/DegreeOfFreedom.cpp DegreesOfFreedom/DegreeOfFreedom.h Primitives/MatrixTest.cpp Primitives/MatrixTest.h)
