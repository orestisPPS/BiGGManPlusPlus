//
// Created by hal9000 on 10/22/23.
//

#ifndef UNTITLED_TRANSIENTDIRICHLET3D_H
#define UNTITLED_TRANSIENTDIRICHLET3D_H

#include <cassert>
#include "../../Analysis/FiniteDifferenceAnalysis/TrasnsientAnalysis/TransientFiniteDifferenceAnalysis.h"
#include "../../StructuredMeshGeneration/MeshFactory.h"
#include "../../StructuredMeshGeneration/MeshSpecs.h"
#include "../../StructuredMeshGeneration/DomainBoundaryFactory.h"
#include "../../LinearAlgebra/Solvers/Iterative/GradientBasedIterative/ConjugateGradientSolver.h"
#include "../../LinearAlgebra/NumericalIntegrators/NewmarkNumericalIntegrator.h"


namespace Tests {

    class TransientNeumann3D{
    public:
        static void runTests(){
            _testTransientNeumann3D();
        }
        
        private:
        static void _testTransientNeumann3D(){
            logTestStart("testDiffusionDirichlet3D");
            map<Direction, unsigned> numberOfNodes;
            numberOfNodes[Direction::One] = 5;
            numberOfNodes[Direction::Two] = 5;
            numberOfNodes[Direction::Three] = 5;

            auto specs = make_shared<MeshSpecs>(numberOfNodes, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0);
            auto meshFactory = new MeshFactory(specs);
            auto meshBoundaries = make_shared<DomainBoundaryFactory>(meshFactory->mesh);
            meshFactory->buildMesh(2, meshBoundaries->parallelepiped(numberOfNodes, 4, 4, 4));
            shared_ptr<Mesh> mesh = meshFactory->mesh;

            auto bottomBC = make_shared<BoundaryCondition>(Dirichlet, shared_ptr<map<DOFType, double>>(
                    new map<DOFType, double>({{Temperature, 500}})));
            auto topBC = make_shared<BoundaryCondition>(Dirichlet, shared_ptr<map<DOFType, double>>(
                    new map<DOFType, double>({{Temperature, 100}})));
            auto leftBC = make_shared<BoundaryCondition>(Dirichlet, shared_ptr<map<DOFType, double>>(
                    new map<DOFType, double>({{Temperature, 20}})));
            auto rightBC = make_shared<BoundaryCondition>(Neumann, shared_ptr<map<DOFType, double>>(
                    new map<DOFType, double>({{Temperature, 0}})));
            auto frontBC = make_shared<BoundaryCondition>(Dirichlet, shared_ptr<map<DOFType, double>>(
                    new map<DOFType, double>({{Temperature, 0}})));
            auto backBC = make_shared<BoundaryCondition>(Dirichlet, shared_ptr<map<DOFType, double>>(
                    new map<DOFType, double>({{Temperature, 0}})));
            
            auto dummyBCMap = make_shared<map<Position, shared_ptr<BoundaryCondition>>>();
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Left, leftBC));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Right, rightBC));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Top, topBC));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Bottom, bottomBC));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Front, frontBC));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Back, backBC));
            auto boundaryConditions = make_shared<DomainBoundaryConditions>(dummyBCMap);

            auto initialConditions = make_shared<InitialConditions>(0, 0);

            auto pdeSpatialProperties = make_shared<SpatialPDEProperties>(3, ScalarField);
            pdeSpatialProperties->setIsotropicSpatialProperties(0.1, 0, 0, 0);

            auto pdeTemporalProperties = make_shared<TransientPDEProperties>(3, ScalarField);
            pdeTemporalProperties->setIsotropicTemporalProperties(0, 0);
            //pdeTemporalProperties->setIsotropicTemporalProperties(0, 0);

            auto heatTransferPDE = make_shared<TransientPartialDifferentialEquation>(pdeSpatialProperties,
                                                                                     pdeTemporalProperties, Laplace);
            auto specsFD = make_shared<FDSchemeSpecs>(2, 2, mesh->directions());
            auto temperatureDOF = new TemperatureScalar_DOFType();
            auto problem = make_shared<TransientMathematicalProblem>(heatTransferPDE, boundaryConditions,
                                                                     initialConditions, temperatureDOF);
            auto solver = make_shared<ConjugateGradientSolver>(1E-9, 1E4, L2, 10);
            auto integration = make_shared<NewmarkNumericalIntegrator>(0.25, 0.5);
            auto initialTime = 0.0;
            auto stepSize = 0.01;
            auto totalSteps = 10;
            auto analysis = make_shared<TransientFiniteDifferenceAnalysis>(initialTime, stepSize, totalSteps, problem, mesh, solver, integration, specsFD);

            auto startAnalysisTime = std::chrono::high_resolution_clock::now();
            analysis->solve();
            auto endAnalysisTime = std::chrono::high_resolution_clock::now();
            cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endAnalysisTime - startAnalysisTime).count() << " ms" << endl;

            auto fileName = "temperature.vtk";
            auto filePath = "/home/hal9000/code/BiGGMan++/Testing/transientVTK/";
            auto fieldType = "Temperature";
            Utility::Exporters::exportTransientScalarFieldResultInVTK(filePath, fileName, fieldType, mesh, totalSteps);

            //auto targetCoords = NumericalVector<double>{0.5, 0.5, 0};
            auto targetCoords = NumericalVector<double>{2, 2, 2};
            auto targetSolution = analysis->getSolutionAtNode(targetCoords, 1E-3);
            targetSolution.printVerticallyWithIndex("Solution");

            auto absoluteRelativeError = abs(targetSolution[targetSolution.size() - 1] - 149.989745970325) / 149.989745970325;

            assert(absoluteRelativeError < 1E-2);
            logTestEnd();
        }
        

        static void logTestStart(const std::string &testName) {
            std::cout << "Running " << testName << "... ";
        }

        static void logTestEnd() {
            std::cout << "\033[1;32m[PASSED ]\033[0m\n";  // This adds a green [PASSED] indicator
        }
    };

} // Tests

#endif //UNTITLED_TRANSIENTDIRICHLET3D_H
