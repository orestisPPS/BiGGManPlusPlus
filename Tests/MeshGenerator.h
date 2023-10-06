//
// Created by hal9000 on 9/15/23.
//

#ifndef UNTITLED_MESHGENERATOR_H
#define UNTITLED_MESHGENERATOR_H
#include "../Analysis/FiniteDifferenceAnalysis/SteadyStateFiniteDifferenceAnalysis.h"
#include "../StructuredMeshGeneration/MeshFactory.h"
#include "../StructuredMeshGeneration/MeshSpecs.h"
#include "../StructuredMeshGeneration/DomainBoundaryFactory.h"
#include "../LinearAlgebra/Solvers/Iterative/GradientBasedIterative/ConjugateGradientSolver.h"

namespace Tests{
    class MeshGenerator {
    public:
        static void buildMesh() {
            map<Direction, unsigned> numberOfNodes;
            numberOfNodes[Direction::One] = 11;
            numberOfNodes[Direction::Two] = 11;
            numberOfNodes[Direction::Three] = 11;

            //auto specs = make_shared<MeshSpecs>(numberOfNodes, 2, 1, 1, 0, 0, 0, 0, 0, 0);
            auto specs = make_shared<MeshSpecs>(numberOfNodes, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0);
            auto meshFactory = make_shared<MeshFactory>(specs);
            auto meshBoundaries = make_shared<DomainBoundaryFactory>(meshFactory->mesh);
            //meshFactory->buildMesh(2, meshBoundaries->annulus_3D_ripGewrgiou(numberOfNodes, 0.1, 1, 0, 45, 2));
            meshFactory->buildMesh(2, meshBoundaries->parallelepiped(numberOfNodes, 4, 4, 4));
            //meshFactory->buildMesh(2, meshBoundaries->parallelepiped(numberOfNodes, 4, 4, 4))

            // 127.83613628736045
            auto bottom = make_shared<BoundaryCondition>(Dirichlet,
                                                         make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                   ({{Temperature,
                                                                                                      100}})));
            auto top = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                           ({{Temperature,
                                                                                                              100}})));
            auto left = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                            ({{Temperature,
                                                                                                               200}})));
            auto right = make_shared<BoundaryCondition>(Dirichlet,
                                                        make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                  ({{Temperature,
                                                                                                     100}})));
            auto front = make_shared<BoundaryCondition>(Dirichlet,
                                                        make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                  ({{Temperature,
                                                                                                     0}})));
            auto back = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                            ({{Temperature,
                                                                                                               500}})));

            shared_ptr<Mesh> mesh = meshFactory->mesh;
            auto fileNameMesh = "meshRebuilt.vtk";
            auto filePathMesh = "/home/hal9000/code/BiGGMan++/Testing/";
            mesh->storeMeshInVTKFile(filePathMesh, fileNameMesh, Natural, false);

            auto pdeProperties = make_shared<SecondOrderLinearPDEProperties>(3);
            pdeProperties->setIsotropicSpatialProperties(10, 0, 0, 0);

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

            auto problem = make_shared<SteadyStateMathematicalProblem>(heatTransferPDE, boundaryConditions,
                                                                       temperatureDOF);

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
            cout << "Target Solution: " << targetSolution[0] << endl;
            auto absoluteRelativeError = abs(targetSolution[0] - 36.6666666666666
            ) / 36.6666666666666;
        }
    };
}
#endif //UNTITLED_MESHGENERATOR_H
