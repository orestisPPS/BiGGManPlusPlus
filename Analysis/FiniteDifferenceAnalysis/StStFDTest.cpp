//
// Created by hal9000 on 3/13/23.
//

#include "StStFDTest.h"
#include "../../Discretization/Mesh/GhostPseudoMesh/GhostPseudoMesh.h"
#include "../../Utility/Exporters/Exporters.h"



namespace NumericalAnalysis {
    
    StStFDTest::StStFDTest() {
        map<Direction, short unsigned> numberOfNodes;
        numberOfNodes[Direction::One] = 5;
        numberOfNodes[Direction::Two] = 5;
        auto specs = new StructuredMeshGenerator::MeshSpecs(numberOfNodes, 2, 1, 0, 10, 10);
        auto space = (PositioningInSpace::Plane);
        auto meshFactory = new MeshFactory(specs);
        auto mesh = meshFactory->mesh;

        auto meshProperties = new SecondOrderLinearPDEProperties(
                2, false, LocallyAnisotropic);
        meshProperties->setLocallyAnisotropicProperties(meshFactory->pdePropertiesFromMetrics);
        auto pde = new PartialDifferentialEquation(meshProperties, GeneralizedSecondOrderLinear);

        auto bcs = createBC();
        auto problem = new SteadyStateMathematicalProblem(pde, bcs, createDOF());
        auto schemeSpecs = createSchemeSpecs();
        auto analysis = new SteadyStateFiniteDifferenceAnalysis(problem, mesh, schemeSpecs);
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
        auto dummyScheme = map<Direction, tuple<FDSchemeType,int>>();
        auto space = PositioningInSpace::Plane;
        return new FDSchemeSpecs(dummyScheme);
    }
    
    
} // NumericalAnalysis