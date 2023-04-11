//
// Created by hal9000 on 3/13/23.
//

#include "StStFDTest.h"
#include "../../Discretization/Mesh/GhostPseudoMesh/GhostPseudoMesh.h"
#include "../../Utility/Exporters/Exporters.h"
#include "../../Discretization/Mesh/GhostPseudoMesh/GhostPseudoMesh2D.h"


namespace NumericalAnalysis {
    
    StStFDTest::StStFDTest() {
        auto mesh = createMesh();
        auto pde = createPDE();
        auto bcs = createBC();
        auto problem = new SteadyStateMathematicalProblem(pde, bcs, createDOF());
        auto schemeSpecs = createSchemeSpecs();
        auto analysis = new SteadyStateFiniteDifferenceAnalysis(problem, mesh, schemeSpecs);
        
        
        auto ghostNodesPerDirection = new map<Direction, unsigned>();
        ghostNodesPerDirection->insert(pair<Direction, unsigned>(Direction::One, 3));
        ghostNodesPerDirection->insert(pair<Direction, unsigned>(Direction::Two, 3));
        ghostNodesPerDirection->insert(pair<Direction, unsigned>(Direction::Two, 0));
        
        auto testGhostMesh =  new GhostPseudoMesh2D(mesh,ghostNodesPerDirection);
        auto filenameParaview = "ghostMesh.vtk";
        auto filePath = "/home/hal9000/code/BiGGMan++/Testing/";
        Utility::Exporters::saveGhostNodesToParaviewFile(testGhostMesh, filePath, filenameParaview);

    }

    Mesh* StStFDTest::createMesh() {
        map<Direction, unsigned> numberOfNodes;
        numberOfNodes[Direction::One] =5;
        numberOfNodes[Direction::Two] = 5;
        auto specs = new StructuredMeshGenerator::MeshSpecs(numberOfNodes, 1, 1,
                                                        0, 10, 10);
        auto space = (PositioningInSpace::Plane);
        auto mesh = StructuredMeshGenerator::MeshFactory(specs).mesh;
        return mesh;            
    }
    
    PartialDifferentialEquation* StStFDTest::createPDE() {
        return new PartialDifferentialEquation(Laplace);
    }
    
    DomainBoundaryConditions* StStFDTest::createBC() {
        auto dummyBCFunctionForAllBoundaryPositions = function<double (vector<double>*)> ([](vector<double>* x) {return 1;});
        auto dummyDOFTypeFunctionMap = new map<DOFType, function<double (vector<double>*)>>();
        
        auto BCDummyMapPair = pair<DOFType, function<double (vector<double>*)>> (Temperature, dummyBCFunctionForAllBoundaryPositions);
        dummyDOFTypeFunctionMap->insert(BCDummyMapPair);
        
        auto leftBC = new BoundaryCondition(BoundaryConditionType::Dirichlet, dummyDOFTypeFunctionMap);
        auto rightBC = new BoundaryCondition(BoundaryConditionType::Dirichlet, dummyDOFTypeFunctionMap);
        auto topBC = new BoundaryCondition(BoundaryConditionType::Dirichlet, dummyDOFTypeFunctionMap);
        auto bottomBC = new BoundaryCondition(BoundaryConditionType::Dirichlet, dummyDOFTypeFunctionMap);
        
        auto dummyBCMap = new map<Position, BoundaryCondition*>();
        dummyBCMap->insert(pair<Position, BoundaryCondition*>(Position::Left, leftBC));
        dummyBCMap->insert(pair<Position, BoundaryCondition*>(Position::Right, rightBC));
        dummyBCMap->insert(pair<Position, BoundaryCondition*>(Position::Top, topBC));
        dummyBCMap->insert(pair<Position, BoundaryCondition*>(Position::Bottom, bottomBC));
        
        return new DomainBoundaryConditions(dummyBCMap);
    }
    
    Field_DOFType* StStFDTest::createDOF() {
        return new TemperatureScalar_DOFType();
        //return new DisplacementVectorField2D_DOFType();
    }
    
    FDSchemeSpecs* StStFDTest::createSchemeSpecs() {
        auto dummyScheme = map<Direction, tuple<FiniteDifferenceSchemeType,int>>();
        auto space = PositioningInSpace::Plane;
        return new FDSchemeSpecs(dummyScheme, space);
    }
    
    
} // NumericalAnalysis