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

    class TransientDirichlet3D {
    public:
        static void runTests(){
            _testTransientDirichlet3D();
        }
        
        private:
        static void _testTransientDirichlet3D(){

            logTestStart("testDiffusionDirichlet3D");logTestStart("testDiffusionDirichlet3D");

            map<Direction, unsigned> numberOfNodes;
            numberOfNodes[Direction::One] = 5;
            numberOfNodes[Direction::Two] = 5;
            numberOfNodes[Direction::Three] = 5;

            //auto specs = make_shared<MeshSpecs>(numberOfNodes, 2, 1, 1, 0, 0, 0, 0, 0, 0);
            auto specs = make_shared<MeshSpecs>(numberOfNodes, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0);
            auto meshFactory = make_shared<MeshFactory>(specs);
            auto meshBoundaries = make_shared<DomainBoundaryFactory>(meshFactory->mesh);
            meshFactory->buildMesh(2, meshBoundaries->parallelepiped(numberOfNodes, 4, 4, 4));
            //meshFactory->buildMesh(2, meshBoundaries->annulus_3D_ripGewrgiou(numberOfNodes, 0.5, 1, 0, 180, 5));
            //meshFactory->mesh->createElements(Hexahedron, 2);
            meshFactory->mesh->storeMeshInVTKFile("/home/hal9000/code/BiGGMan++/Testing/", "threeDeeMeshBoi.vtk", Natural, false);

            // 127.83613628736045
            auto bottom = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                              ({{Temperature, 100}})));
            auto top = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                           ({{Temperature, 100}})));
            auto left = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                            ({{Temperature, 20}})));
            auto right = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                             ({{Temperature, 0}})));
            auto front = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                             ({{Temperature, 0}})));
            auto back = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                            ({{Temperature, 0}})));


            auto dummyBCMap = make_shared<map<Position, shared_ptr<BoundaryCondition>>>();
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Left, left));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Right, right));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Top, top));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Bottom, bottom));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Front, front));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Back, back));

            auto boundaryConditions = make_shared<DomainBoundaryConditions>(dummyBCMap);

            auto initialConditions = make_shared<InitialConditions>(0, 0);
            
            shared_ptr<Mesh> mesh = meshFactory->mesh;

            auto pdeSpatialProperties = make_shared<SpatialPDEProperties>(3, ScalarField);
            pdeSpatialProperties->setIsotropicSpatialProperties(10, 0, 0, 0);
            
            auto pdeTemporalProperties = make_shared<TransientPDEProperties>(3, ScalarField);
            pdeTemporalProperties->setIsotropicTemporalProperties(0, -1);

            auto heatTransferPDE = make_shared<TransientPartialDifferentialEquation>(pdeSpatialProperties, pdeTemporalProperties);

            auto specsFD = make_shared<FDSchemeSpecs>(2, 2, mesh->directions());
            
            auto temperatureDOF = new TemperatureScalar_DOFType();

            auto problem = make_shared<TransientMathematicalProblem>(heatTransferPDE, boundaryConditions, initialConditions, temperatureDOF);

            auto solver = make_shared<ConjugateGradientSolver>(1E-9, 1E4, L2, 10);
            auto integration = make_shared<NewmarkNumericalIntegrator>(0.25, 0.5);
            auto initialTime = 0.0;
            auto stepSize = 1;
            auto totalSteps = 20;
            auto analysis = make_shared<TransientFiniteDifferenceAnalysis>(initialTime, stepSize, totalSteps, problem, mesh, solver, integration, specsFD);

            auto startAnalysisTime = std::chrono::high_resolution_clock::now();
            analysis->solve();
            auto endAnalysisTime = std::chrono::high_resolution_clock::now();
            cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endAnalysisTime - startAnalysisTime).count() << " ms" << endl;

            auto fileName = "temperature.vtk";
            auto filePath = "/home/hal9000/code/BiGGMan++/Testing/transientVTK/";
            auto fieldType = "Temperature";
            Utility::Exporters::exportTransientScalarFieldResultInVTK(filePath, fileName, fieldType, mesh, totalSteps);

            analysis->solve();

            analysis->applySolutionToDegreesOfFreedom();
            
            auto targetCoords = NumericalVector<double>{2, 2, 2};
            //auto targetCoords = NumericalVector<double>{1.5, 1.5, 3};
            //auto targetCoords = NumericalVector<double>{5, 5, 5};
            auto targetSolution = analysis->getSolutionAtNode(targetCoords, 1E-1);
            cout<<"Computed Value : "<< targetSolution[0] << endl;
            cout<<"Expected Solution: "<< 36.6666666666666 << endl;
            auto absoluteRelativeError = abs(targetSolution[0] - 36.6666666666666
            ) / 36.6666666666666;

            assert(absoluteRelativeError < 1E-12);
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
