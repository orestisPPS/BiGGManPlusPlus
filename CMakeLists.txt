cmake_minimum_required(VERSION 3.23)
project(untitled LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -fopenmp)


enable_language(CUDA)


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
        LinearAlgebra/Array/Array.h
        LinearAlgebra/Operations/TransformationTensors.h 
        Utility/Calculators.h

        LinearAlgebra/Operations/Transformations.cpp
        LinearAlgebra/Operations/Transformations.h
        PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h
        LinearAlgebra/FiniteDifferences/FiniteDifferenceSchemeWeightsStructuredGrid.cpp
        LinearAlgebra/FiniteDifferences/FiniteDifferenceSchemeWeightsStructuredGrid.h
        LinearAlgebra/FiniteDifferences/FDSchemeSpecs.cpp
        LinearAlgebra/FiniteDifferences/FDSchemeSpecs.h
        Analysis/FiniteDifferenceAnalysis/FiniteDifferenceAnalysis.h
        Discretization/Node/NodalCoordinates.h
        Discretization/Node/NodalCoordinates.cpp
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
        LinearAlgebra/AnalysisLinearSystemInitializer.h
        LinearAlgebra/AnalysisLinearSystemInitializer.cpp
        Discretization/Node/IsoparametricNodeGraph.h
        Discretization/Node/IsoparametricNodeGraph.cpp
        Utility/Exporters/Exporters.h Utility/Exporters/Exporters.cpp
        LinearAlgebra/FiniteDifferences/FiniteDifferenceSchemeBuilder.cpp
        Discretization/Mesh/GhostPseudoMesh/GhostPseudoMesh.h
        Discretization/Mesh/GhostPseudoMesh/GhostPseudoMesh.cpp 
        Discretization/Mesh/Metrics/Metrics.h
        Discretization/Mesh/Metrics/Metrics.cpp
        LinearAlgebra/Operations/VectorOperations.h
        LinearAlgebra/Solvers/Solver.h LinearAlgebra/Solvers/Solver.cpp 
        LinearAlgebra/Norms/VectorNorm.h LinearAlgebra/Norms/VectorNorm.cpp
        LinearAlgebra/Solvers/Iterative/IterativeSolver.h 
        LinearAlgebra/Solvers/Iterative/IterativeSolver.cpp
        LinearAlgebra/FiniteDifferences/FDSchemeType.h
        LinearAlgebra/Solvers/Direct/SolverLUP.h LinearAlgebra/Solvers/Direct/SolverLUP.cpp
        LinearAlgebra/Solvers/Direct/DirectSolver.h
        LinearAlgebra/Solvers/Direct/DirectSolver.cpp
        LinearAlgebra/LinearSystem.h
        LinearAlgebra/LinearSystem.cpp
        LinearAlgebra/Solvers/Direct/Cholesky.h
        LinearAlgebra/Solvers/Direct/Cholesky.cpp
        LinearAlgebra/Array/DecompositionMethods/DecompositionLUP.h
        LinearAlgebra/Array/DecompositionMethods/DecompositionLUP.cpp
        LinearAlgebra/Array/DecompositionMethods/MatrixDecomposition.cpp
        LinearAlgebra/FiniteDifferences/FDWeightCalculator.h 
        PartialDifferentialEquations/SpaceFieldProperties.h 
        StructuredMeshGeneration/DomainBoundaryFactory.h
        StructuredMeshGeneration/DomainBoundaryFactory.cpp
        LinearAlgebra/AnalysisLinearSystemInitializer2.h 
        LinearAlgebra/AnalysisLinearSystemInitializer2.cpp
        LinearAlgebra/Solvers/Iterative/StationaryIterative/JacobiSolver.h 
        LinearAlgebra/Solvers/Iterative/StationaryIterative/JacobiSolver.cpp 
        LinearAlgebra/Solvers/Iterative/StationaryIterative/GaussSeidelSolver.h
        LinearAlgebra/Solvers/Iterative/StationaryIterative/GaussSeidelSolver.cpp
        LinearAlgebra/Solvers/Iterative/StationaryIterative/StationaryIterative.h 
        LinearAlgebra/Solvers/Iterative/StationaryIterative/StationaryIterative.cpp 
        LinearAlgebra/Solvers/Iterative/StationaryIterative/SORSolver.h 
        LinearAlgebra/Solvers/Iterative/StationaryIterative/SORSolver.cpp 
        Tests/SteadyState3DNeumann.h Tests/SteadyState3DNeumann.cpp 
        Discretization/Elements/Element.h 
        Discretization/Elements/Element.cpp
        Discretization/Elements/MeshElements.h
        Discretization/Elements/MeshElements.cpp
        LinearAlgebra/Solvers/Iterative/StationaryIterative/StationaryIterrativeCuda/StationaryIterativeCuda.cuh
        LinearAlgebra/Solvers/Iterative/StationaryIterative/StationaryIterrativeCuda/StationaryIterativeCuda.cu
        LinearAlgebra/Norms/VectorNormCuda.cuh
        LinearAlgebra/Norms/VectorNormCuda.cu LinearAlgebra/Solvers/Iterative/StationaryIterative/StationaryIterrativeCuda/JacobiCuda.cuh LinearAlgebra/Solvers/Iterative/StationaryIterative/StationaryIterrativeCuda/JacobiCuda.cu LinearAlgebraCUDA/NumericalOperationsCUDA.cuh LinearAlgebraCUDA/NumericalOperationsCUDA.cu LinearAlgebraCUDA/MemoryManagementCUDA.cuh LinearAlgebraCUDA/MemoryManagementCUDA.cu Tests/OperationsCUDA.h Tests/OperationsCUDA.cpp LinearAlgebra/Array/NumericalArray.h LinearAlgebra/Array/NumericalArray.cpp LinearAlgebra/Solvers/Iterative/GradientBasedIterative/ConjugateGradientSolver.h LinearAlgebra/Solvers/Iterative/GradientBasedIterative/ConjugateGradientSolver.cpp
        LinearAlgebra/Operations/MultiThreadVectorOperations.h
        LinearAlgebra/Operations/MultiThreadVectorOperations.cpp
        Tests/VectorOperationsTest.h
        Tests/VectorOperationsTest.cpp
        Tests/SteadyStateDirichlet3D.h
        Tests/SteadyStateDirichlet3D.cpp
        Tests/Mesh3dTest.h
        Tests/Mesh3dTest.cpp
        LinearAlgebra/EigenDecomposition/LanczosEigenDecomposition.h
        LinearAlgebra/EigenDecomposition/LanczosEigenDecomposition.cpp
        Tests/LanczosEigenDecompositionTest.h
        Tests/LanczosEigenDecompositionTest.cpp)


target_link_libraries(untitled cudart)

