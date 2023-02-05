cmake_minimum_required(VERSION 3.23)
project(untitled)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -fopenmp)

add_executable(untitled main.cpp
        DegreesOfFreedom/DegreeOfFreedom.cpp 
        DegreesOfFreedom/DegreeOfFreedom.h
        LinearAlgebra/MatrixTest.cpp
        LinearAlgebra/MatrixTest.h
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
        StructuredMeshGeneration/MeshTest2D.h
        LinearAlgebra/Array.h
        LinearAlgebra/TransformationTensors.h 
        Utility/Calculators.h
        LinearAlgebra/Transformations.cpp
        LinearAlgebra/Transformations.h
        PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.cpp
        PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h
        PositioningInSpace/CoordinateVector.cpp
        PositioningInSpace/CoordinateVector.h
        Discretization/Node/NodeNeighbourFinder.cpp
        Discretization/Node/NodeNeighbourFinder.h
        LinearAlgebra/FiniteDifferences/FirstOrderDerivativeFDScheme.cpp
        LinearAlgebra/FiniteDifferences/FirstOrderDerivativeFDScheme.h
        LinearAlgebra/FiniteDifferences/SecondOrderDerivativeFDScheme.cpp
        LinearAlgebra/FiniteDifferences/SecondOrderDerivativeFDScheme.h 
        LinearAlgebra/FiniteDifferences/FiniteDifferenceSchemeCoefficientCalculator.cpp
        LinearAlgebra/FiniteDifferences/FiniteDifferenceSchemeCoefficientCalculator.h LinearAlgebra/FiniteDifferences/FDSchemeSpecs.cpp LinearAlgebra/FiniteDifferences/FDSchemeSpecs.h FiniteDifferenceAnalysis/FiniteDifferenceAnalysis.h)
        
        