cmake_minimum_required(VERSION 3.23)
project(untitled)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -fopenmp)

add_executable(untitled main.cpp DegreesOfFreedom/DegreeOfFreedom.cpp DegreesOfFreedom/DegreeOfFreedom.h Primitives/MatrixTest.cpp Primitives/MatrixTest.h PositioningInSpace/Coordinate.cpp PositioningInSpace/Coordinate.h PositioningInSpace/Discretization/Node.cpp PositioningInSpace/Discretization/Node.h Discretization/DiscreteEntityId.cpp Discretization/DiscreteEntityId.h)
