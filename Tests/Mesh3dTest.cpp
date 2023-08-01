/*
//
// Created by hal9000 on 7/30/23.
//

#include "Mesh3dTest.h"

namespace Tests {
    
        Mesh3dTest::Mesh3dTest() {
            map<Direction, unsigned> numberOfNodes;
            numberOfNodes[Direction::One] = 5;
            numberOfNodes[Direction::Two] = 5;
            numberOfNodes[Direction::Three] = 5;

            //auto specs = make_shared<MeshSpecs>(numberOfNodes, 2, 1, 1, 0, 0, 0, 0, 0, 0);
            auto specs = make_shared<MeshSpecs>(numberOfNodes, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0);
            auto meshFactory = new MeshFactory(specs);
            auto meshBoundaries = make_shared<DomainBoundaryFactory>(meshFactory->mesh);
            meshFactory->buildMesh(2, meshBoundaries->parallelepiped(numberOfNodes, 4, 4, 4));
            _mesh = meshFactory->mesh;
        }
    
        void Mesh3dTest::runTestForID() {
            cout << "Mesh ID: " << _mesh->getID() << endl;
        }
    
        void Mesh3dTest::runTestForParametricCoordinates() {
            cout << "Mesh parametric coordinates: " << endl;
            for (auto &parametricCoordinate : _mesh->getParametricCoordinates()) {
                cout << parametricCoordinate << endl;
            }
        }
    
        void Mesh3dTest::runTestForSolutionWithJacobiSolver_Sequential() {
            auto analysis = SteadyStateFiniteDifferenceAnalysis(_mesh, SolverType::JacobiSolver, ParallelizationMethod::Sequential);
            analysis.solve();
            analysis.printSolution();
        }
    
        void Mesh3dTest::runTestForSolutionWithJacobiSolver_MultiThreaded() {
            auto analysis = SteadyStateFiniteDifferenceAnalysis(_mesh, SolverType::JacobiSolver, ParallelizationMethod::MultiThreaded);
            analysis.solve();
            analysis.printSolution();
        }
    
        void Mesh3dTest::runTestForSolutionWithJacobiSolver_Cuda() {
            auto analysis = SteadyStateFiniteDifferenceAnalysis(_mesh, SolverType::JacobiSolver, ParallelizationMethod::Cuda);
            analysis.solve();
            analysis.printSolution();
        }
    
        void Mesh3dTest::runTestForSolutionWithGaussSeidelSolver_Sequential() {
            auto analysis = SteadyStateFiniteDifferenceAnalysis(_mesh, SolverType::GaussSeidelSolver, ParallelizationMethod::Sequential);
            analysis.solve();
            analysis.printSolution();
        }
    
        void Mesh3dTest::runTestForSolutionWithGaussSeidelSolver_MultiThreaded() {
            auto analysis = SteadyStateFiniteDifferenceAnalysis(_mesh, SolverType::GaussSeidelSolver, ParallelizationMethod::MultiThreaded);
            analysis.solve();
            analysis.printSolution();
        }
    
        void Mesh3dTest::runTestForSolutionWithGaussSeidelSolver_Cuda() {
            auto analysis = SteadyStateFiniteDifferenceAnalysis(_mesh, SolverType::GaussSeidelSolver, ParallelizationMethod::Cuda);
            analysis.solve();
            analysis.printSolution();
        }
    
        void Mesh3d
} // Tests*/
