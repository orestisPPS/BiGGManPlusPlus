/*
//
// Created by hal9000 on 8/3/23.
//

#include "LanczosEigenDecompositionTest.h"
#include "../LinearAlgebra/EigenDecomposition/QR/GramSchmidtQR.h"
#include "../LinearAlgebra/EigenDecomposition/QR/IterationQR.h"

namespace Tests {
    LanczosEigenDecompositionTest::LanczosEigenDecompositionTest() {
        
        map<Direction, unsigned> numberOfNodes;
        numberOfNodes[Direction::One] = 5;
        numberOfNodes[Direction::Two] = 4;
        numberOfNodes[Direction::Three] = 4;

        //auto specs = make_shared<MeshSpecs>(numberOfNodes, 2, 1, 1, 0, 0, 0, 0, 0, 0);
        auto specs = make_shared<MeshSpecs>(numberOfNodes, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0, 0, 0);
        auto meshFactory = make_shared<MeshFactory>(specs);
        auto meshBoundaries = make_shared<DomainBoundaryFactory>(meshFactory->mesh);
        meshFactory->buildMesh(2, meshBoundaries->parallelepiped(numberOfNodes, 1, 1, 1));
        //meshFactory->buildMesh(2, meshBoundaries->annulus_3D_ripGewrgiou(numberOfNodes, 0.5, 1, 0, 180, 5));
        //meshFactory->mesh->createElements(Hexahedron, 2);
        meshFactory->mesh->storeMeshInVTKFile("/home/hal9000/code/BiGGMan++/Testing/", "threeDeeMeshBoi.vtk", Natural, false);

        // 127.83613628736045
        auto bottom = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                ({{Temperature, 10}})));
        auto top = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                ({{Temperature, 5}})));
        auto left = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                ({{Temperature, 2}})));
        auto right = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                ({{Temperature, 0}})));
        auto front = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                ({{Temperature, 0}})));
        auto back = make_shared<BoundaryCondition>(Dirichlet, make_shared<map<DOFType, double>>(map<DOFType, double>
                ({{Temperature, 0}})));

        shared_ptr<Mesh> mesh = meshFactory->mesh;

        auto pdeProperties = make_shared<SecondOrderLinearPDEProperties>(3, false, Isotropic);
        pdeProperties->setIsotropicProperties(1,0,0,0);

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

        auto matrixToDecompose = analysis->linearSystem->matrix;
        //auto eigenDecomposition = make_shared<LanczosEigenDecomposition>(10, 8);
        //auto eigenDecomposition = make_shared<PowerMethod>(10, 100);
        //eigenDecomposition->setMatrix(analysis->linearSystem->matrix);
        //auto eigenDecomposition = make_shared<DecompositionQR>(analysis->linearSystem->matrix);
        //auto qr = make_shared<LinearAlgebra::GramSchmidtQR>(analysis->linearSystem->matrix);
        //qr->decompose();
        //eigenDecomposition->calculateEigenvalues();
        //eigenDecomposition->calculateDominantEigenValue();

        auto qr = make_shared<IterationQR>(10, 1E-4, Householder, SingleThread, true);
        qr->setMatrix(analysis->linearSystem->matrix);
        qr->calculateEigenvalues();
        Utility::Exporters::exportLinearSystemToMatlabFile(analysis->linearSystem->matrix, analysis->linearSystem->rhs,"/home/hal9000/code/BiGGMan++/Testing/", "linearSystemEigen2.m", false);
        
        analysis->solve();
        

        analysis->applySolutionToDegreesOfFreedom();

        auto fileName = "temperatureField.vtk";
        auto filePath = "/home/hal9000/code/BiGGMan++/Testing/";
        auto fieldType = "Temperature";
        Utility::Exporters::exportScalarFieldResultInVTK(filePath, fileName, fieldType, analysis->mesh);

        auto targetCoords = NumericalVector<double>{0.5, 0.5, 0.5};
        //auto targetCoords = NumericalVector<double>{1.5, 1.5, 1.5};
        //auto targetCoords = NumericalVector<double>{2, 2, 2};
        //auto targetCoords = NumericalVector<double>{1.5, 1.5, 3};
        //auto targetCoords = NumericalVector<double>{5, 5, 5};
        auto targetSolution = analysis->getSolutionAtNode(targetCoords, 1E-1);
        cout<<"Target Solution: "<< targetSolution[0] << endl;



        auto filenameParaview = "firstMesh.vtk";
    }
} // Tests*/
