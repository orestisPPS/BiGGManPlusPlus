//
// Created by hal9000 on 7/15/23.
//

#include "SteadyState3DNeumann.h"
#include "../LinearAlgebra/Solvers/Iterative/StationaryIterative/SORSolver.h"

namespace Tests {
    SteadyState3DNeumann::SteadyState3DNeumann() {
        map<Direction, unsigned> numberOfNodes;
        numberOfNodes[Direction::One] = 5;
        numberOfNodes[Direction::Two] = 5;
        numberOfNodes[Direction::Three] = 5;

        auto specs = make_shared<MeshSpecs>(numberOfNodes, 1, 1, 1, 0, 0, 0, 0, 0, 0);
        auto meshFactory = new MeshFactory(specs);
        auto meshBoundaries = make_shared<DomainBoundaryFactory>(meshFactory->mesh);
        meshFactory->buildMesh(2, meshBoundaries->parallelepiped(numberOfNodes, 4, 4, 1));
        //meshFactory->buildMesh(2, meshBoundaries->annulus_3D_ripGewrgiou(numberOfNodes, 0.1, 1, 0, 180, 3));
        meshFactory->mesh->storeMeshInVTKFile("/home/hal9000/code/BiGGMan++/Testing/", "threeDeeMeshBoi.vtk", Natural);
/*        for (auto& node : *meshFactory->mesh->boundaryNodes->at(Position::Bottom)) {
            meshFactory->mesh->getNormalUnitVectorOfBoundaryNode(Position::Bottom, node);
        }*/


        
        auto bottomBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 100}}));
        auto topBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 100}}));
        auto rightBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 50}}));
        auto leftBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 50}}));
        auto frontBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 1000}}));
        auto backBC = new BoundaryCondition(Neumann, new map<DOFType, double>(
                {{Temperature, 0}}));

        shared_ptr<Mesh> mesh = meshFactory->mesh;

        auto pdeProperties = make_shared<SecondOrderLinearPDEProperties>(3, false, Isotropic);
        pdeProperties->setIsotropicProperties(1,0,0,0);

        auto heatTransferPDE = make_shared<PartialDifferentialEquation>(pdeProperties, Laplace);

        auto specsFD = make_shared<FDSchemeSpecs>(2, 2, mesh->directions());

        auto dummyBCMap = make_shared<map<Position, shared_ptr<BoundaryCondition>>>();
        dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Left, leftBC));
        dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Right, rightBC));
        dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Top, topBC));
        dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Bottom, bottomBC));
        dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Front, frontBC));
        dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Back, backBC));
        
        auto boundaryConditions = make_shared<DomainBoundaryConditions>(dummyBCMap);
        
        auto temperatureDOF = new TemperatureScalar_DOFType();
        
        auto problem = make_shared<SteadyStateMathematicalProblem>(heatTransferPDE, boundaryConditions, temperatureDOF);
        
        //auto solver = new SolverLUP(1E-20, true);//
        //auto solver  = new JacobiSolver(false, VectorNormType::LInf);
        //auto solver  = new GaussSeidelSolver(true, VectorNormType::LInf, 1E-9);
        auto solver = make_shared<SORSolver>(1.8, true, VectorNormType::L2, 1E-10);
        auto analysis = new SteadyStateFiniteDifferenceAnalysis(problem, mesh, solver, specsFD);
        
        analysis->solve();
        
        analysis->applySolutionToDegreesOfFreedom();
        
        //auto targetCoords = vector<double>{0.5, 0.5};
        auto targetCoords = vector<double>{2, 0, 2};
        auto targetSolution = analysis->getSolutionAtNode(targetCoords, 1E-4);
        
        cout<<"Target Solution: "<< targetSolution[0] << endl;

        auto fileName = "temperatureField.vtk";
        auto filePath = "/home/hal9000/code/BiGGMan++/Testing/";
        auto fieldType = "Temperature";
        Utility::Exporters::exportScalarFieldResultInVTK(filePath, fileName, fieldType, analysis->mesh);
        
        auto filenameParaview = "firstMesh.vtk";
    }
} // Tests