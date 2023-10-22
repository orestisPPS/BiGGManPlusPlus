//
// Created by hal9000 on 10/20/23.
//

#ifndef UNTITLED_TRANSIENTDIRICHLET2D_H
#define UNTITLED_TRANSIENTDIRICHLET2D_H


#include <cassert>
#include "../../Analysis/FiniteDifferenceAnalysis/TrasnsientAnalysis/TransientFiniteDifferenceAnalysis.h"
#include "../../StructuredMeshGeneration/MeshFactory.h"
#include "../../StructuredMeshGeneration/MeshSpecs.h"
#include "../../StructuredMeshGeneration/DomainBoundaryFactory.h"
#include "../../LinearAlgebra/Solvers/Iterative/GradientBasedIterative/ConjugateGradientSolver.h"
#include "../../LinearAlgebra/NumericalIntegrators/NewmarkNumericalIntegrator.h"

namespace Tests {

    class TransientDirichlet2D {
    public:
        static void runTests() {
            _testTransientDiffusionDirichlet2D();
        }

    private:

        static void _testTransientDiffusionDirichlet2D() {
            logTestStart("testDiffusionDirichlet2D");

            map<Direction, unsigned> numberOfNodes;
            numberOfNodes[Direction::One] = 5;
            numberOfNodes[Direction::Two] = 5;

            auto specs = make_shared<MeshSpecs>(numberOfNodes, 1, 1, 0, 0, 0);
            auto meshFactory = make_shared<MeshFactory>(specs);
            auto meshBoundaries = make_shared<DomainBoundaryFactory>(meshFactory->mesh);
            meshFactory->buildMesh(2, meshBoundaries->parallelogram(numberOfNodes, 1, 1, 0, 0));
            shared_ptr<Mesh> mesh = meshFactory->mesh;

/*            auto fileNameMesh = "test_2D_dirichlet.vtk";
            auto filePathMesh = "/home/hal9000/code/BiGGMan++/Testing/";
            mesh->storeMeshInVTKFile(filePathMesh, fileNameMesh, Natural, false);*/
            // 127.83613628736045

            auto bottom = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                              ({{Temperature, 100}})));
            auto top = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                           ({{Temperature, 500}})));
            auto left = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                            ({{Temperature, 20}})));
            auto right = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                                                                                                             ({{Temperature, 0}})));
            auto dummyBCMap = make_shared<map<Position, shared_ptr<BoundaryCondition>>>();
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Left, left));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Right, right));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Top, top));
            dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Bottom, bottom));
            auto boundaryConditions = make_shared<DomainBoundaryConditions>(dummyBCMap);

            auto initialConditions = make_shared<InitialConditions>(0, 0);


            auto pdeSpatialProperties = make_shared<SpatialPDEProperties>(3, ScalarField);
            pdeSpatialProperties->setIsotropicSpatialProperties(0.1, 0, 0, 0);

            auto pdeTemporalProperties = make_shared<TransientPDEProperties>(3, ScalarField);
            pdeTemporalProperties->setIsotropicTemporalProperties(0, -1);
            //pdeTemporalProperties->setIsotropicTemporalProperties(0, 0);

            auto heatTransferPDE = make_shared<TransientPartialDifferentialEquation>(pdeSpatialProperties,
                                                                                     pdeTemporalProperties, Laplace);

            auto specsFD = make_shared<FDSchemeSpecs>(2, 2, mesh->directions());

            auto temperatureDOF = new TemperatureScalar_DOFType();
            auto problem = make_shared<TransientMathematicalProblem>(heatTransferPDE, boundaryConditions,
                                                                     initialConditions, temperatureDOF);

            auto solver = make_shared<ConjugateGradientSolver>(1E-9, 1E4, L2, 10);
            auto integration = make_shared<NewmarkNumericalIntegrator>(0.25, 0.5);
            auto analysis = make_shared<TransientFiniteDifferenceAnalysis>(0.0, 0.1, 20, problem, mesh, solver, integration, specsFD);

            //analysis->linearSystem->exportToMatlabFile("transientDiffusionLinearSystem.m",
                                                     //  "/home/hal9000/code/BiGGMan++/Testing/", true);

            analysis->solve();

            analysis->applySolutionToDegreesOfFreedom();

            auto fileName = "temperatureField.vtk";
            auto filePath = "/home/hal9000/code/BiGGMan++/Testing/";
            auto fieldType = "Temperature";
            Utility::Exporters::exportScalarFieldResultInVTK(filePath, fileName, fieldType, analysis->mesh);

            auto targetCoords = NumericalVector<double>{0.5, 0.5, 0};

            auto targetSolution = analysis->getSolutionAtNode(targetCoords, 1E-3);

            auto absoluteRelativeError = abs(targetSolution[0] - 149.989745970325) / 149.989745970325;

            assert(absoluteRelativeError < 1E-3);
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
    

#endif //UNTITLED_TRANSIENTDIRICHLET2D_H