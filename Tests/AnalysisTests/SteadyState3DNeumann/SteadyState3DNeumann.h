//
// Created by hal9000 on 7/15/23.
//

#ifndef UNTITLED_STEADYSTATE3DNEUMANN_H
#define UNTITLED_STEADYSTATE3DNEUMANN_H

#include <cassert>
#include "../../../Analysis/FiniteDifferenceAnalysis/SteadyStateFiniteDifferenceAnalysis.h"
#include "../../../StructuredMeshGeneration/MeshFactory.h"
#include "../../../StructuredMeshGeneration/MeshSpecs.h"
#include "../../../StructuredMeshGeneration/DomainBoundaryFactory.h"
#include "../../../LinearAlgebra/Solvers/Iterative/GradientBasedIterative/ConjugateGradientSolver.h"


namespace Tests {

    class SteadyState3DNeumann {
    public:
        static void runTests(){
            //_testDiffusionDirichlet3D();
            _testDiffusionNeumann3D();
            
        }
        
        private:
        static void _testDiffusionDirichlet3D() {
            
            logTestStart("testDiffusionDirichlet3D");

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
            
            shared_ptr<Mesh> mesh = meshFactory->mesh;
            auto fileNameMesh = "meshRebuilt.vtk";
            auto filePathMesh = "/home/hal9000/code/BiGGMan++/Testing/";
            mesh->storeMeshInVTKFile(filePathMesh, fileNameMesh, Natural, false);

            auto heatTransferPDE = make_shared<PartialDifferentialEquation>(ScalarField, 3);
            heatTransferPDE->spatialDerivativesCoefficients()->setIsotropic(10, 0, 0, 0);
            
            auto specsFD = make_shared<FDSchemeSpecs>(2, 2, mesh->directions());
            
            auto temperatureDOF = new TemperatureScalar_DOFType();


            // 127.83613628736045
            auto boundaryConditions = make_shared<DomainBoundaryConditions>();
            boundaryConditions->setBoundaryCondition({Right, Front, Back}, Dirichlet, Temperature, 0.0);
            boundaryConditions->setBoundaryCondition({Bottom, Top}, Dirichlet, Temperature, 100.0);
            boundaryConditions->setBoundaryCondition({Left}, Dirichlet, Temperature, 20.0);
            
            auto problem = make_shared<SteadyStateMathematicalProblem>(heatTransferPDE, boundaryConditions, temperatureDOF);

            // auto solver = make_shared<SolverLUP>(1E-20, true);
            //auto solver = make_shared<JacobiSolver>(VectorNormType::L2, 1E-10, 1E4, true, vTechKickInYoo);
            //auto solver = make_shared<GaussSeidelSolver>(turboVTechKickInYoo, VectorNormType::LInf, 1E-9);
            //auto solver = make_shared<GaussSeidelSolver>(VectorNormType::L2, 1E-9, 1E4, false, turboVTechKickInYoo);
            auto solver = make_shared<ConjugateGradientSolver>(1E-12, 1E4, L2);
            //auto solver = make_shared<SORSolver>(1.8, VectorNormType::L2, 1E-5);
            auto analysis = make_shared<SteadyStateFiniteDifferenceAnalysis>(problem, mesh, solver, specsFD);

            analysis->solve();

            analysis->applySolutionToDegreesOfFreedom();

            auto fileName = "temperatureField.vtk";
            auto filePath = "/home/hal9000/code/BiGGMan++/Testing/";
            auto fieldType = "Temperature";
            Utility::Exporters::exportScalarFieldResultInVTK(filePath, fileName, fieldType, analysis->mesh);

            //auto targetCoords = NumericalVector<double>{0.5, 0.5};
            //auto targetCoords = NumericalVector<double>{1.5, 1.5, 1.5};
            auto targetCoords = NumericalVector<double>{2, 2, 2};
            //auto targetCoords = NumericalVector<double>{1.5, 1.5, 3};
            //auto targetCoords = NumericalVector<double>{5, 5, 5};
            auto targetSolution = analysis->getSolutionAtNode(targetCoords, 1E-1);
            cout<<"Computed Value : "<< targetSolution[0] << endl;
            auto absoluteRelativeError = abs(targetSolution[0] - 36.6666666666666
            ) / 36.6666666666666;

            assert(absoluteRelativeError< 1E-12);
            logTestEnd();

            auto filenameParaview = "firstMesh.vtk";
        }

        static void _testDiffusionNeumann3D() {

            logTestStart("testDiffusionNeumann3D");

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
                                                                                                              ({{Temperature, 1000}})));
            auto top = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                           ({{Temperature, 0}})));
            auto left = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                            ({{Temperature, 5}})));
            auto right = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                             ({{Temperature, 0}})));
            auto front = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                             ({{Temperature, 0}})));
            auto back = make_shared<BoundaryCondition>(Neumann, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                            ({{Temperature, 0}})));

            shared_ptr<Mesh> mesh = meshFactory->mesh;
            auto fileNameMesh = "meshRebuilt.vtk";
            auto filePathMesh = "/home/hal9000/code/BiGGMan++/Testing/";
            mesh->storeMeshInVTKFile(filePathMesh, fileNameMesh, Natural, false);

            auto pdeProperties = make_shared<SpatialPDEProperties>(3, ScalarField);
            pdeProperties->setIsotropic(10, 0, 0, 0);

            auto heatTransferPDE = make_shared<PartialDifferentialEquation>(pdeProperties, Laplace);

            auto specsFD = make_shared<FDSchemeSpecs>(2, 2, mesh->directions());

            auto dummyBCMap = make_shared<map<Position, shared_ptr<BoundaryCondition>>>();
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Left, left));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Right, right));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Top, top));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Bottom, bottom));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Front, front));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Back, back));

            auto boundaryConditions = make_shared<DomainBoundaryConditions>(dummyBCMap);

            auto temperatureDOF = new TemperatureScalar_DOFType();

            auto problem = make_shared<SteadyStateMathematicalProblem>(heatTransferPDE, boundaryConditions, temperatureDOF);

            // auto solver = make_shared<SolverLUP>(1E-20, true);
            //auto solver = make_shared<JacobiSolver>(VectorNormType::L2, 1E-10, 1E4, true, vTechKickInYoo);
            //auto solver = make_shared<GaussSeidelSolver>(turboVTechKickInYoo, VectorNormType::LInf, 1E-9);
            //auto solver = make_shared<GaussSeidelSolver>(VectorNormType::L2, 1E-9, 1E4, false, turboVTechKickInYoo);
            auto solver = make_shared<ConjugateGradientSolver>(1E-12, 1E4, L2);
            //auto solver = make_shared<SORSolver>(1.8, VectorNormType::L2, 1E-5);
            auto analysis = make_shared<SteadyStateFiniteDifferenceAnalysis>(problem, mesh, solver, specsFD);

            analysis->solve();

            analysis->applySolutionToDegreesOfFreedom();

            auto fileName = "temperatureField.vtk";
            auto filePath = "/home/hal9000/code/BiGGMan++/Testing/";
            auto fieldType = "Temperature";
            Utility::Exporters::exportScalarFieldResultInVTK(filePath, fileName, fieldType, analysis->mesh);

            //auto targetCoords = NumericalVector<double>{0.5, 0.5};
            //auto targetCoords = NumericalVector<double>{1.5, 1.5, 1.5};
            auto targetCoords = NumericalVector<double>{2, 2, 2};
            //auto targetCoords = NumericalVector<double>{1.5, 1.5, 3};
            //auto targetCoords = NumericalVector<double>{5, 5, 5};
            auto targetSolution = analysis->getSolutionAtNode(targetCoords, 1E-3);
            cout<<"Computed Value : "<< targetSolution[0] << endl;
            cout<<"Expected Solution: "<< 106.710125311661 << endl;
            auto absoluteRelativeError = abs(targetSolution[0] - 106.710125311661
            ) / 106.710125311661;

            assert(absoluteRelativeError < 1E-12);
            logTestEnd();

            auto filenameParaview = "firstMesh.vtk";
        }

        static void logTestStart(const std::string& testName) {
            std::cout << "Running " << testName << "... ";
        }

        static void logTestEnd() {
            std::cout << "\033[1;32m[PASSED ]\033[0m\n";  // This adds a green [PASSED] indicator
        }
    };
    

} // Tests

#endif //UNTITLED_STEADYSTATE3DNEUMANN_H
