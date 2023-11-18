//
// Created by hal9000 on 10/20/23.
//

#ifndef UNTITLED_TRANSIENTDIRICHLET2D_H
#define UNTITLED_TRANSIENTDIRICHLET2D_H


#include <cassert>
#include "../../../Analysis/FiniteDifferenceAnalysis/TrasnsientAnalysis/TransientFiniteDifferenceAnalysis.h"
#include "../../../StructuredMeshGeneration/MeshFactory.h"
#include "../../../StructuredMeshGeneration/MeshSpecs.h"
#include "../../../StructuredMeshGeneration/DomainBoundaryFactory.h"
#include "../../../LinearAlgebra/Solvers/Iterative/GradientBasedIterative/ConjugateGradientSolver.h"
#include "../../../LinearAlgebra/NumericalIntegrators/NewmarkNumericalIntegrator.h"

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
            numberOfNodes[Direction::One] = 11;
            numberOfNodes[Direction::Two] = 11;

            auto specs = make_shared<MeshSpecs>(numberOfNodes, 1, 1, 0, 0, 0);
            auto meshFactory = make_shared<MeshFactory>(specs);
            auto meshBoundaries = make_shared<DomainBoundaryFactory>(meshFactory->mesh);
            meshFactory->buildMesh(2, meshBoundaries->parallelogram(numberOfNodes, 1, 1, 0, 0));
            shared_ptr<Mesh> mesh = meshFactory->mesh;

            auto pde = make_shared<PartialDifferentialEquation>(ScalarField, 2, true);
            pde->spatialDerivativesCoefficients()->setIsotropic(0.1, 0, 0, 0);
            pde->temporalDerivativesCoefficients()->setIsotropic(0, -1);
            
            auto boundaryConditions = make_shared<DomainBoundaryConditions>();
            boundaryConditions->setBoundaryCondition(Bottom, Dirichlet, Temperature, 1);
            boundaryConditions->setBoundaryCondition(Top, Dirichlet, Temperature, 5);
            boundaryConditions->setBoundaryCondition(Left, Dirichlet, Temperature, 2);
            boundaryConditions->setBoundaryCondition(Right, Dirichlet, Temperature, 0);
            
            auto initialConditions = make_shared<InitialConditions>(0, 0);

            auto specsFD = make_shared<FDSchemeSpecs>(2, 2, mesh->directions());
            auto temperatureDOF = new TemperatureScalar_DOFType();
            auto problem = make_shared<TransientMathematicalProblem>(pde, boundaryConditions, initialConditions, temperatureDOF);
            auto solver = make_shared<ConjugateGradientSolver>(1E-9, 1E4, L2, 10);
            auto integration = make_shared<NewmarkNumericalIntegrator>(0.25, 0.5);
            auto initialTime = 0.0;
            auto stepSize = 1;
            auto totalSteps = 30;
            auto analysis = make_shared<TransientFiniteDifferenceAnalysis>(initialTime, stepSize, totalSteps, problem, mesh,
                                                                           solver, integration, specsFD);
            
            auto startAnalysisTime = std::chrono::high_resolution_clock::now();
            analysis->solve();
            auto endAnalysisTime = std::chrono::high_resolution_clock::now();
            cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endAnalysisTime - startAnalysisTime).count() << " ms" << endl;
            
            auto fileName = "temperature.vtk";
            auto filePath = "/home/hal9000/code/BiGGMan++/Tests/AnalysisTests/Transient2DDirichlet";
            auto fieldType = "Temperature";
            Utility::Exporters::exportTransientScalarFieldResultInVTK(filePath, fileName, fieldType, mesh, totalSteps);

            auto targetCoords = NumericalVector<double>{0.5, 0.5, 0};
            //auto targetCoords = NumericalVector<double>{2, 2, 0};
            auto targetSolution = analysis->getSolutionAtNode(targetCoords, 1E-3);
            targetSolution.exportCSV(filePath, "biggMannSolution");
            targetSolution.printVerticallyWithIndex("Solution");
            //0.3279032790882641,5.959866049560129,23.327234434676296,43.39089546228296,62.08687503609156,78.11091674856848,91.75732178444765,103.14608999939946,112.54633691428205,120.26233357134024,126.58035882987187,131.7492042549288,135.9771601315341,139.43572273709907,142.26519893515157,144.58019316462156,146.47433266468704,148.02415293847162,149.29224932201706,150.3298313589199
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
    

#endif //UNTITLED_TRANSIENTDIRICHLET2D_H