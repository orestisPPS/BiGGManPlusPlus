//
// Created by hal9000 on 3/13/23.
//

#include "StStFDTest.h"
#include "../../Discretization/Mesh/GhostPseudoMesh/GhostPseudoMesh.h"
#include "../../Utility/Exporters/Exporters.h"



namespace NumericalAnalysis {
    
    StStFDTest::StStFDTest() {
        map<Direction, unsigned> numberOfNodes;
        numberOfNodes[Direction::One] = 11;
        numberOfNodes[Direction::Two] = 11;
        auto specs = new MeshSpecs(numberOfNodes, 1, 1, 0, 0, 0);
        auto meshFactory = new MeshFactory(specs);
        meshFactory->domainBoundaryFactory->parallelogram(numberOfNodes, 1, 1);
        //meshFactory->domainBoundaryFactory->ellipse(numberOfNodes, 1, 1);
        meshFactory->buildMesh(2);
        
        auto pdeProperties =
                new SecondOrderLinearPDEProperties(2, false, Isotropic);
        pdeProperties->setIsotropicProperties(1,0,0,0);
        
        auto heatTransferPDE = new PartialDifferentialEquation(pdeProperties, Laplace);
        
        auto specsFD = new FDSchemeSpecs(2, 2, meshFactory->mesh->directions());


        auto bottomBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                                                                        {{Temperature, 200}}));
        auto topBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                                                                        {{Temperature, 100}}));
        auto rightBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                                                                        {{Temperature, 0}}));
        auto leftBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                                                                        {{Temperature, 0}}));
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
                new SteadyStateFiniteDifferenceAnalysis(problem, meshFactory->mesh, solver, specsFD, Parametric);
        
        analysis->solve();
        
        analysis->applySolutionToDegreesOfFreedom();
        
        auto targetCoords = vector<double>{0.5, 0.5};
        auto targetSolution = analysis->getSolutionAtNode(targetCoords);
        
/*        auto result = analysis->linearSystem->solution;
        
        for (double i : *result) {
            cout << i << endl;
        }*/
        
        auto filenameParaview = "firstMesh.vtk";
        auto path = "/home/hal9000/code/BiGGMan++/Testing/";
        auto mesh = meshFactory->mesh;
        mesh->storeMeshInVTKFile(path, filenameParaview);
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