//
// Created by hal9000 on 3/13/23.
//

#include "StStFDTest.h"
#include "../../Discretization/Mesh/GhostPseudoMesh/GhostPseudoMesh.h"
#include "../../Utility/Exporters/Exporters.h"



namespace NumericalAnalysis {
    
    StStFDTest::StStFDTest() {
        map<Direction, unsigned> numberOfNodes;
        numberOfNodes[Direction::One] = 50;
        numberOfNodes[Direction::Two] = 11;
        auto specs = new MeshSpecs(numberOfNodes, 1, 1, 0, 0, 0);
        auto meshFactory = new MeshFactory(specs);
        //meshFactory->domainBoundaryFactory->parallelogram(numberOfNodes, 1, 1);
        //meshFactory->domainBoundaryFactory->ellipse(numberOfNodes, 1, 1);
        meshFactory->domainBoundaryFactory->ripGewrgiou(numberOfNodes, 0.1, 1, M_PI_2f64x / 2, 2 * M_PI_2f64x);
        meshFactory->buildMesh(2);
        
        meshFactory->mesh->storeMeshInVTKFile("/home/hal9000/code/BiGGMan++/Testing/", "meshEllipse.vtk");
        
        Mesh* mesh = meshFactory->mesh;
        
        auto pdeProperties =
                new SecondOrderLinearPDEProperties(2, false, Isotropic);
        pdeProperties->setIsotropicProperties(1,0,0,0);
        
        auto heatTransferPDE = new PartialDifferentialEquation(pdeProperties, Laplace);
        
        auto specsFD = new FDSchemeSpecs(2, 2, mesh->directions());

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
        
        //--------------------------------COMSOL TEST---------------------------------------------------------------------
        auto bottomBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 0}}));
        auto topBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 0}}));
        auto rightBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 500 }}));
        auto leftBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 0}}));
        
        
        //--------------------------------https://kyleniemeyer.github.io/ME373-book/content/pdes/elliptic.html---------------------------------------------------------------------
        //-------------------------------------------------------------------------Passes-----------------------------
/*        auto bottomBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 100}}));
        auto topBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 00}}));
        auto rightBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 100}}));
        auto leftBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 100}}));*/
        
        //--------------------------Valougeorgis examples10 pg. 6 (0 Dirichlet, -1 Source)-----------------------------
        //--------------------------Passes-----------------------------
/*        auto pdeProperties =
                new SecondOrderLinearPDEProperties(2, false, Isotropic);
        pdeProperties->setIsotropicProperties(1,0,0,-1);
        
        auto bottomBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 0}}));
        auto topBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 0}}));
        auto rightBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 0}}));
        auto leftBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                {{Temperature, 0}}));*/

        auto dummyBCMap = new map<Position, BoundaryCondition*>();
        dummyBCMap->insert(pair<Position, BoundaryCondition*>(Position::Left, leftBC));
        dummyBCMap->insert(pair<Position, BoundaryCondition*>(Position::Right, rightBC));
        dummyBCMap->insert(pair<Position, BoundaryCondition*>(Position::Top, topBC));
        dummyBCMap->insert(pair<Position, BoundaryCondition*>(Position::Bottom, bottomBC));
        
        auto boundaryConditions = new DomainBoundaryConditions(dummyBCMap);
        
        auto temperatureDOF = new TemperatureScalar_DOFType();
        
        auto problem = new SteadyStateMathematicalProblem(heatTransferPDE, boundaryConditions, temperatureDOF);
        
        auto solver = new SolverLUP(1E-20, true);
        
        auto analysis =
                new SteadyStateFiniteDifferenceAnalysis(problem, mesh, solver, specsFD);
        
        

        
        //analysis->solve();
        
        //analysis->applySolutionToDegreesOfFreedom();


        
        auto targetCoords = vector<double>{0.5, 0.5};
        //auto targetCoords = vector<double>{2, 2};
        auto targetSolution = analysis->getSolutionAtNode(targetCoords, 1E-2);
        
        cout<<"Target Solution: "<< targetSolution[0] << endl;

        auto fileName = "temperatureField.vtk";
        auto filePath = "/home/hal9000/code/BiGGMan++/Testing/";
        auto fieldType = "Temperature";
        Utility::Exporters::exportScalarFieldResultInVTK(filePath, fileName, fieldType, analysis->mesh);
        
        auto filenameParaview = "firstMesh.vtk";
    }


    
    Field_DOFType* StStFDTest::createDOF() {
        //return new TemperatureScalar_DOFType();
        return new nodalPositionVectorField2D_DOFType();
        //return new DisplacementVectorField2D_DOFType();
    }
    
    FDSchemeSpecs* StStFDTest::createSchemeSpecs() {

        return new FDSchemeSpecs(2, 2, {One, Two});
    }
    
    
} // NumericalAnalysis