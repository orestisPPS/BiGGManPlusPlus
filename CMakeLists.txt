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
        PartialDifferentialEquations/SecondOrderLinearPDEProperties.cpp 
        PartialDifferentialEquations/SecondOrderLinearPDEProperties.h 
        PartialDifferentialEquations/HeatTransfer/SteadyStateHeatTransferEquation.cpp
        PartialDifferentialEquations/HeatTransfer/EnergyTransferEquation.h
        MathematicalProblem/SteadyStateMathematicalProblem.cpp
        MathematicalProblem/SteadyStateMathematicalProblem.h
        MathematicalProblem/TransientMathematicalProblem.cpp
        MathematicalProblem/TransientMathematicalProblem.h
        StructuredMeshGeneration/MeshFactory.cpp
        StructuredMeshGeneration/MeshFactory.h
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
        PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h
        PositioningInSpace/CoordinateVector.cpp
        PositioningInSpace/CoordinateVector.h
        LinearAlgebra/FiniteDifferences/FirstOrderDerivativeFDSchemeCalculator.cpp
        LinearAlgebra/FiniteDifferences/FirstOrderDerivativeFDSchemeCalculator.h
        LinearAlgebra/FiniteDifferences/FDSchemeSpecs.cpp
        LinearAlgebra/FiniteDifferences/FDSchemeSpecs.h
        Analysis/FiniteDifferenceAnalysis/FiniteDifferenceAnalysis.h
        Discretization/Node/NodalCoordinates.h
        Discretization/Node/NodalCoordinates.cpp
        DegreesOfFreedom/DegreeOfFreedomID.cpp
        Analysis/FiniteDifferenceAnalysis/FiniteDifferenceAnalysis.cpp
        BoundaryConditions/DomainBoundaryConditions.h
        BoundaryConditions/DomainBoundaryConditions.cpp
        BoundaryConditions/BoundaryCondition.cpp
        Analysis/AnalysisDOFs/DOFInitializer.h
        Analysis/AnalysisDOFs/DOFInitializer.cpp
        DegreesOfFreedom/DegreeOfFreedomTypes.h
        PartialDifferentialEquations/PartialDifferentialEquation.cpp
        Utility/EnumerablesTranformations.h
        Discretization/Mesh/Mesh1D.h
        Discretization/Mesh/Mesh1D.cpp 
        Discretization/Mesh/Mesh2D.h
        Discretization/Mesh/Mesh2D.cpp
        Discretization/Mesh/Mesh3D.h
        Discretization/Mesh/Mesh3D.cpp
        Analysis/AnalysisDOFs/AnalysisDegreesOfFreedom.h
        Analysis/AnalysisDOFs/AnalysisDegreesOfFreedom.cpp
        Analysis/NumericalAnalysis.h Analysis/NumericalAnalysis.cpp
        MathematicalProblem/MathematicalProblem.h
        MathematicalProblem/MathematicalProblem.cpp
        Analysis/FiniteDifferenceAnalysis/SteadyStateFiniteDifferenceAnalysis.h 
        Analysis/FiniteDifferenceAnalysis/SteadyStateFiniteDifferenceAnalysis.cpp
        Analysis/FiniteDifferenceAnalysis/StStFDTest.h
        Analysis/FiniteDifferenceAnalysis/StStFDTest.cpp
        DegreesOfFreedom/DegreesOfFreedomCategorization.h
        DegreesOfFreedom/DegreesOfFreedomCategorization.cpp
        LinearAlgebra/LinearSystem.h 
        LinearAlgebra/LinearSystem.cpp
        Discretization/Node/IsoparametricNodeGraph.h Discretization/Node/IsoparametricNodeGraph.cpp Utility/Exporters/Exporters.h Utility/Exporters/Exporters.cpp LinearAlgebra/FiniteDifferences/FiniteDifferenceSchemeBuilder.cpp Discretization/Mesh/GhostPseudoMesh/GhostPseudoMesh.h Discretization/Mesh/GhostPseudoMesh/GhostPseudoMesh.cpp Discretization/Mesh/Metrics/Metrics.h Discretization/Mesh/Metrics/Metrics.cpp LinearAlgebra/VectorOperations.h LinearAlgebra/VectorOperations.cpp LinearAlgebra/FiniteDifferences/FDSchemeComponent.h LinearAlgebra/FiniteDifferences/FDSchemeComponent.cpp LinearAlgebra/Solvers/Solver.h LinearAlgebra/Solvers/Solver.cpp LinearAlgebra/Norms/VectorNorm.h LinearAlgebra/Norms/VectorNorm.cpp LinearAlgebra/Solvers/Iterative/IterativeSolver.h LinearAlgebra/Solvers/Iterative/IterativeSolver.cpp)
        
        