//
// Created by hal9000 on 3/13/23.
//

#include "StStFDTest.h"
#include "../../Discretization/Mesh/GhostPseudoMesh/GhostPseudoMesh.h"
#include "../../Utility/Exporters/Exporters.h"
#include "../../LinearAlgebra/Solvers/Iterative/StationaryIterative/JacobiSolver.h"
#include "../../LinearAlgebra/Solvers/Iterative/StationaryIterative/GaussSeidelSolver.h"
#include "../../LinearAlgebra/Solvers/Iterative/StationaryIterative/SORSolver.h"


namespace NumericalAnalysis {
    
    StStFDTest::StStFDTest() {
        
        /*map<Direction, unsigned> numberOfNodes;
        numberOfNodes[Direction::One] = 5;
        numberOfNodes[Direction::Two] = 5;
        auto specs = make_shared<MeshSpecs>(numberOfNodes, 1, 1, 0, 0, 0);
        auto meshFactory = new MeshFactory(specs);
        meshFactory->domainBoundaryFactory->parallelogram(numberOfNodes, 4, 4);
        //meshFactory->domainBoundaryFactory->ellipse(numberOfNodes, 1, 1);
        //meshFactory->domainBoundaryFactory->annulus_ripGewrgiou(numberOfNodes, 0.5, 1, 0, 270);
        //meshFactory->domainBoundaryFactory->cavityBot(numberOfNodes, 1, 1);
        //meshFactory->domainBoundaryFactory->gasTankHorizontal(numberOfNodes, 1, 1);
        //cout<<"yo"<<endl;
        //meshFactory->domainBoundaryFactory->sinusRiver(numberOfNodes, 1.5, 1, 0.1, 4);
        meshFactory->buildMesh(2);
        
        meshFactory->mesh->storeMeshInVTKFile("/home/hal9000/code/BiGGMan++/Testing/", "meshEllipse.vtk");*/
        map<Direction, unsigned> numberOfNodes;
        numberOfNodes[Direction::One] = 11;
        numberOfNodes[Direction::Two] = 11;
        numberOfNodes[Direction::Three] = 3;
        auto specs = make_shared<MeshSpecs>(numberOfNodes, 1, 1, 0.25, 0, 0, 0, 0, 0, 0);
        auto meshFactory = new MeshFactory(specs);
        meshFactory->domainBoundaryFactory->parallelepiped(numberOfNodes, 4, 4, 1);
        meshFactory->buildMesh(2);
        meshFactory->mesh->storeMeshInVTKFile("/home/hal9000/code/BiGGMan++/Testing/", "threeDeeMeshBoi.vtk", Natural);

        //meshFactory->domainBoundaryFactory->parallelogram(numberOfNodes, 5, 5);
        //meshFactory->domainBoundaryFactory->ellipse(numberOfNodes, 1, 1);
        //meshFactory->domainBoundaryFactory->annulus_ripGewrgiou(numberOfNodes, 0.5, 1, 0, 270);
        //meshFactory->domainBoundaryFactory->cavityBot(numberOfNodes, 1, 1);
        //meshFactory->domainBoundaryFactory->gasTankHorizontal(numberOfNodes, 1, 1);
        //cout<<"yo"<<endl;
        //meshFactory->domainBoundaryFactory->sinusRiver(numberOfNodes, 1.5, 1, 0.1, 4);
        //meshFactory->buildMesh(2);

        //meshFactory->mesh->storeMeshInVTKFile("/home/hal9000/code/BiGGMan++/Testing/", "meshEllipse.vtk");
        

        //--------------------------------CHAPRA PG 857---------------------------------------------------------------------
        //--------------------------Passes-----------------------------
/*
        auto bottomBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                                                                        {{Temperature, 0}}));
        auto topBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                                                                        {{Temperature, 100}}));
        auto rightBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                                                                        {{Temperature, 50}}));
        auto leftBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                                                                        {{Temperature, 75}}));*/
        
/*        //--------------------------------COMSOL TEST---------------------------------------------------------------------
        auto bottomBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 0}}));
        auto topBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 0}}));
        auto rightBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 500 }}));
        auto leftBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 0}}));
        
*//*        
        //--------------------------------https://kyleniemeyer.github.io/ME373-book/content/pdes/elliptic.html---------------------------------------------------------------------
        //-------------------------------------------------------------------------Passes-----------------------------
        auto bottomBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 100}}));
        auto topBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 00}}));
        auto rightBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 100}}));
        auto leftBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 100}}));*//*
        
*//*        //--------------------------Valougeorgis examples10 pg. 6 (0 Dirichlet, -1 Source)-----------------------------
        //--------------------------Passes-----------------------------
        auto pdeProperties =
                new SecondOrderLinearPDEProperties(2, false, Isotropic);
        pdeProperties->setIsotropicProperties(1,0,0,-1);
        
        auto bottomBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 0}}));
        auto topBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 0}}));
        auto rightBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 0}}));
        auto leftBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 0}}));*//*

        shared_ptr<Mesh> mesh = meshFactory->mesh;

        auto pdeProperties = make_shared<SecondOrderLinearPDEProperties>(2, false, Isotropic);
        pdeProperties->setIsotropicProperties(1,0,0,0);

        auto heatTransferPDE = make_shared<PartialDifferentialEquation>(pdeProperties, Laplace);

        auto specsFD = make_shared<FDSchemeSpecs>(2, 2, mesh->directions());

        auto dummyBCMap = make_shared<map<Position, shared_ptr<BoundaryCondition>>>();
        dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Left, leftBC));
        dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Right, rightBC));
        dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Top, topBC));
        dummyBCMap->insert(pair<Position, shared_ptr<BoundaryCondition>>(Position::Bottom, bottomBC));
        
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
        auto targetCoords = vector<double>{2, 2};
        auto targetSolution = analysis->getSolutionAtNode(targetCoords, 1E-6);
        
        cout<<"Target Solution: "<< targetSolution[0] << endl;

        auto fileName = "temperatureField.vtk";
        auto filePath = "/home/hal9000/code/BiGGMan++/Testing/";
        auto fieldType = "Temperature";
        Utility::Exporters::exportScalarFieldResultInVTK(filePath, fileName, fieldType, analysis->mesh);
        
        auto filenameParaview = "firstMesh.vtk";*/
    }
    
    
} // NumericalAnalysis