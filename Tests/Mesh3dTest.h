/*
//
// Created by hal9000 on 7/30/23.
//

#ifndef UNTITLED_MESH3DTEST_H
#define UNTITLED_MESH3DTEST_H

#include "../Analysis/FiniteDifferenceAnalysis/SteadyStateFiniteDifferenceAnalysis.h"
#include "../StructuredMeshGeneration/MeshFactory.h"
#include "../StructuredMeshGeneration/MeshSpecs.h"
#include "../StructuredMeshGeneration/DomainBoundaryFactory.h"
#include "../LinearAlgebra/Solvers/Direct/SolverLUP.h"
#include "../LinearAlgebra/Solvers/Iterative/StationaryIterative/SORSolver.h"
#include "../LinearAlgebra/Solvers/Iterative/GradientBasedIterative/ConjugateGradientSolver.h"

namespace Tests {

    class Mesh3dTest {
    public:
        Mesh3dTest();
        
        void runTestForID();
        
        void runTestForParametricCoordinates();

        void runTestForSolutionWithJacobiSolver_Sequential();

        void runTestForSolutionWithJacobiSolver_MultiThreaded();

        void runTestForSolutionWithJacobiSolver_Cuda();
        
        void runTestForSolutionWithGaussSeidelSolver_Sequential();
        
        void runTestForSolutionWithGaussSeidelSolver_MultiThreaded();
        
        void runTestForSolutionWithGaussSeidelSolver_Cuda();
        
        void runTestForSolutionWithSORSolver_Sequential();
        
        void runTestForSolutionWithSORSolver_MultiThreaded();
        
        void runTestForSolutionWithSORSolver_Cuda();
        
        void runTestForSolutionWithConjugateGradientSolver_Sequential();
        
        void runTestForSolutionWithConjugateGradientSolver_MultiThreaded();
        
        void runTestForSolutionWithConjugateGradientSolver_Cuda();
        
    private:
        shared_ptr<Mesh> _mesh;
        
        
    };

} // Tests

#endif //UNTITLED_MESH3DTEST_H
*/
