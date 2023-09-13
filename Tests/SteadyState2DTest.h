//
// Created by hal9000 on 9/12/23.
//

#ifndef UNTITLED_STEADYSTATE2DTEST_H
#define UNTITLED_STEADYSTATE2DTEST_H

#endif //UNTITLED_STEADYSTATE2DTEST_H

#include <cassert>
#include "../Analysis/FiniteDifferenceAnalysis/SteadyStateFiniteDifferenceAnalysis.h"
#include "../StructuredMeshGeneration/MeshFactory.h"
#include "../StructuredMeshGeneration/MeshSpecs.h"
#include "../StructuredMeshGeneration/DomainBoundaryFactory.h"
#include "../LinearAlgebra/Solvers/Iterative/GradientBasedIterative/ConjugateGradientSolver.h"

namespace Tests {

    class SteadyState2DTest {
    public:
        static void runTests(){
            //_testDiffusionDirichlet2D();
            _testDiffusionDirichlet2D2();
            
        }
        
    private:
        static void _testDiffusionDirichlet2D(){
            logTestStart("testDiffusionDirichlet2D");

            map<Direction, unsigned> numberOfNodes;
            numberOfNodes[Direction::One] = 5;
            numberOfNodes[Direction::Two] = 5;

            auto specs = make_shared<MeshSpecs>(numberOfNodes, 1, 1, 0, 0, 0);
            auto meshFactory = make_shared<MeshFactory>(specs);
            auto meshBoundaries = make_shared<DomainBoundaryFactory>(meshFactory->mesh);
            meshFactory->buildMesh(2, meshBoundaries->parallelogram(numberOfNodes, 4, 4, 0, 0));
            shared_ptr<Mesh> mesh = meshFactory->mesh;
            
/*            auto fileNameMesh = "test_2D_dirichlet.vtk";
            auto filePathMesh = "/home/hal9000/code/BiGGMan++/Testing/";
            mesh->storeMeshInVTKFile(filePathMesh, fileNameMesh, Natural, false);*/
            // 127.83613628736045
            
            auto bottom = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                      ({{Temperature, 100}})));
            auto top = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                        ({{Temperature, 100}})));
            auto left = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                        ({{Temperature, 20}})));
            auto right = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                         ({{Temperature, 0}})));



            auto pdeProperties = make_shared<SecondOrderLinearPDEProperties>(3, false, Isotropic);
            pdeProperties->setIsotropicProperties(10,0,0,0);

            auto heatTransferPDE = make_shared<PartialDifferentialEquation>(pdeProperties, Laplace);

            auto specsFD = make_shared<FDSchemeSpecs>(2, 2, mesh->directions());

            auto dummyBCMap = make_shared<map<Position, shared_ptr<BoundaryCondition>>>();
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Left, left));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Right, right));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Top, top));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Bottom, bottom));
           

            auto boundaryConditions = make_shared<DomainBoundaryConditions>(dummyBCMap);
            auto temperatureDOF = new TemperatureScalar_DOFType();
            auto problem = make_shared<SteadyStateMathematicalProblem>(heatTransferPDE, boundaryConditions, temperatureDOF);
            auto solver = make_shared<ConjugateGradientSolver>(1E-9, 1E4, L2);
            auto analysis = make_shared<SteadyStateFiniteDifferenceAnalysis>(problem, mesh, solver, specsFD);

            analysis->solve();

            analysis->applySolutionToDegreesOfFreedom();

            auto fileName = "temperatureField.vtk";
            auto filePath = "/home/hal9000/code/BiGGMan++/Testing/";
            auto fieldType = "Temperature";
            Utility::Exporters::exportScalarFieldResultInVTK(filePath, fileName, fieldType, analysis->mesh);
            
            auto targetCoords = NumericalVector<double>{2, 2, 0};

            auto targetSolution = analysis->getSolutionAtNode(targetCoords, 1E-3);
            
            auto absoluteRelativeError = abs(targetSolution[0] - 55) / 55;

            assert(absoluteRelativeError - 55 < 1E-12);
            logTestEnd();
        }

        static void _testDiffusionDirichlet2D2(){
            logTestStart("testDiffusionDirichlet2D");

            map<Direction, unsigned> numberOfNodes;
            numberOfNodes[Direction::One] = 5;
            numberOfNodes[Direction::Two] = 11;
            //numberOfNodes[Direction::Three] = 1;

            //auto specs = make_shared<MeshSpecs>(numberOfNodes, 0.1, 0.1, 0, 0, 0);
            auto specs = make_shared<MeshSpecs>(numberOfNodes, 2, 1, 0, 0, 0);
            auto meshFactory = make_shared<MeshFactory>(specs);
            auto meshBoundaries = make_shared<DomainBoundaryFactory>(meshFactory->mesh);
            meshFactory->buildMesh(2, meshBoundaries->parallelogram(numberOfNodes, 8, 4, 0, 0));
            auto mesh = meshFactory->mesh;

/*            auto fileNameMesh = "test_2D_dirichlet.vtk";
            auto filePathMesh = "/home/hal9000/code/BiGGMan++/Testing/";
            mesh->storeMeshInVTKFile(filePathMesh, fileNameMesh, Natural, false);*/
            // 127.83613628736045

            auto bottom = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                              ({{Temperature, 100}})));
            auto top = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                           ({{Temperature, 100}})));
            auto left = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                            ({{Temperature, 20}})));
            auto right = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                             ({{Temperature, 0}})));



            auto pdeProperties = make_shared<SecondOrderLinearPDEProperties>(2, false, Isotropic);
            pdeProperties->setIsotropicProperties(10,0,0,0);

            auto heatTransferPDE = make_shared<PartialDifferentialEquation>(pdeProperties, Laplace);

            auto specsFD = make_shared<FDSchemeSpecs>(2, 2, mesh->directions());

            auto dummyBCMap = make_shared<map<Position, shared_ptr<BoundaryCondition>>>();
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Left, left));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Right, right));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Top, top));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Bottom, bottom));


            auto boundaryConditions = make_shared<DomainBoundaryConditions>(dummyBCMap);
            auto temperatureDOF = new TemperatureScalar_DOFType();
            auto problem = make_shared<SteadyStateMathematicalProblem>(heatTransferPDE, boundaryConditions, temperatureDOF);
            auto solver = make_shared<ConjugateGradientSolver>(1E-12, 1E4, L2, true);
            auto analysis = make_shared<SteadyStateFiniteDifferenceAnalysis>(problem, mesh, solver, specsFD);

            analysis->solve();

            analysis->applySolutionToDegreesOfFreedom();

            auto fileName = "temperatureField.vtk";
            auto filePath = "/home/hal9000/code/BiGGMan++/Testing/";
            auto fieldType = "Temperature";
            Utility::Exporters::exportScalarFieldResultInVTK(filePath, fileName, fieldType, analysis->mesh);

            //auto targetCoords = NumericalVector<double>{5, 5, 0};
            //auto targetCoords = NumericalVector<double>{2, 2, 0};
            auto targetCoords = NumericalVector<double>{2, 1, 0};

            auto targetSolution = analysis->getSolutionAtNode(targetCoords, 1E-3);
            cout << "Target Solution: " << targetSolution[0] << endl;
            auto absoluteRelativeError = abs(targetSolution[0] - 55) / 55;

            assert(absoluteRelativeError < 1E-3);
            logTestEnd();
        }

        static void logTestStart(const std::string& testName) {
            std::cout << "Running " << testName << "... ";
        }

        static void logTestEnd() {
            std::cout << "\033[1;32m[PASSED ]\033[0m\n";  // This adds a green [PASSED] indicator
        }
    };

} // Tests