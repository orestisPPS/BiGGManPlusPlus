cmake_minimum_required(VERSION 3.23)
project(untitled)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -fopenmp)

add_executable(untitled main.cpp
        DegreesOfFreedom/DegreeOfFreedom.cpp 
        DegreesOfFreedom/DegreeOfFreedom.h
        LinearAlgebra/MatrixTest.cpp
        LinearAlgebra/MatrixTest.h 
        PositioningInSpace/Coordinate.cpp
        PositioningInSpace/Coordinate.h
        Discretization/Node/Node.cpp
        Discretization/Node/Node.h
        Discretization/Id/DiscreteEntityId.cpp
        Discretization/Id/DiscreteEntityId.h 
        BoundaryConditions/BoundaryCondition.cpp
        BoundaryConditions/BoundaryCondition.h
        PartialDifferentialEquations/SecondOrderLinearPDEProperties.cpp 
        PartialDifferentialEquations/SecondOrderLinearPDEProperties.h 
        PartialDifferentialEquations/HeatTransfer/SteadyStateHeatTransferEquation.cpp
        PartialDifferentialEquations/HeatTransfer/EnergyTransferEquation.h
        MathematicalProblem/SteadyStateMathematicalProblem.cpp
        MathematicalProblem/SteadyStateMathematicalProblem.h
        MathematicalProblem/TransientMathematicalProblem.cpp
        MathematicalProblem/TransientMathematicalProblem.h
        StructuredMeshGeneration/MeshPreProcessor2D.cpp
        StructuredMeshGeneration/MeshPreProcessor.h
        StructuredMeshGeneration/MeshSpecs.cpp
        StructuredMeshGeneration/MeshSpecs.h
        Discretization/Mesh/Mesh.cpp
        Discretization/Mesh/Mesh.h
        StructuredMeshGeneration/NodeFactory.cpp
        StructuredMeshGeneration/NodeFactory.h
        StructuredMeshGeneration/MeshTest2D.cpp
        StructuredMeshGeneration/MeshTest2D.h LinearAlgebra/Array.h PositioningInSpace/PhysicalSpaceCharacteristics.cpp PositioningInSpace/SpaceCharacteristics.h)
