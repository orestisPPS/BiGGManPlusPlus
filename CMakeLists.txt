cmake_minimum_required(VERSION 3.23)
project(untitled)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -fopenmp)

add_executable(untitled main.cpp DegreesOfFreedom/DegreeOfFreedom.cpp 
        DegreesOfFreedom/DegreeOfFreedom.h 
        Primitives/MatrixTest.cpp
        Primitives/MatrixTest.h 
        PositioningInSpace/Coordinate.cpp
        PositioningInSpace/Coordinate.h 
        Discretization/Node.cpp
        Discretization/Node.h
        Discretization/DiscreteEntityId.cpp
        Discretization/DiscreteEntityId.h 
        BoundaryConditions/BoundaryCondition.cpp
        BoundaryConditions/BoundaryCondition.h 
        PartialDifferentialEquations/DifferentialEquation.cpp 
        PartialDifferentialEquations/DifferentialEquation.h)
