//
// Created by hal9000 on 3/13/23.
//

#include "StStFDTest.h"
#include "../../Discretization/Mesh/GhostPseudoMesh/GhostPseudoMesh.h"
#include "../../Utility/Exporters/Exporters.h"



namespace NumericalAnalysis {
    
    StStFDTest::StStFDTest() {
        map<Direction, short unsigned> numberOfNodes;
/*        numberOfNodes[Direction::One] = 5;
        numberOfNodes[Direction::Two] = 7;*/
        numberOfNodes[Direction::One] = 5;
        numberOfNodes[Direction::Two] = 5;
        //auto specs = new StructuredMeshGenerator::MeshSpecs(numberOfNodes, 0.25, 0.25, 0, 0, 0);
        auto specs = new StructuredMeshGenerator::MeshSpecs(numberOfNodes, 1, 1, 0, 0, 0);
        auto space = (PositioningInSpace::Plane);
        auto meshFactory = new MeshFactory(specs);
        auto mesh = meshFactory->mesh;

        auto meshProperties = new SecondOrderLinearPDEProperties(
                2, false, LocallyAnisotropic);
        meshProperties->setLocallyAnisotropicProperties(meshFactory->pdePropertiesFromMetrics);
        
/*        auto laplaceProperties = new SecondOrderLinearPDEProperties(
                2, false, Isotropic);
        laplaceProperties->setIsotropicProperties(1, 0, 0, 0);*/
        
        auto pde = new PartialDifferentialEquation(meshProperties, Laplace);

        auto bcs = createBC(mesh);
        auto problem = new SteadyStateMathematicalProblem(pde, bcs, createDOF());
        auto schemeSpecs = createSchemeSpecs();
        auto solver = new SolverLUP(1E-20, true);
        auto analysis = new SteadyStateFiniteDifferenceAnalysis(problem, mesh, solver, schemeSpecs);
        analysis->solve();
        auto result = analysis->linearSystem->solution;
        for (double i : *result) {
            cout << i << endl;
        }
    }

    DomainBoundaryConditions* StStFDTest::createBC(Mesh *mesh) {
        
        auto boundaryFactory = new DomainBoundaryFactory(mesh);
        return boundaryFactory->parallelogram(5, 5, 4, 4);

/*        auto bcFunctionBottom = function<double (vector<double>*)> ([](vector<double>* x) {return  x->at(0) * x->at(0);} );
        auto bcFunctionTop= function<double (vector<double>*)> ([](vector<double>* x) {return  x->at(0) * x->at(0) - 1;} );
        auto bcFunctionRight = function<double (vector<double>*)> ([](vector<double>* x) {return 1 - x->at(1) * x->at(1);} );
        auto bcFunctionLeft = function<double (vector<double>*)> ([](vector<double>* x) {return  - x->at(1) * x->at(1);} );*/
        /*
        auto bottomBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                                                                        {{Temperature, 0}}));
        auto topBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                                                                        {{Temperature, 75}}));
        auto rightBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                                                                        {{Temperature, 0}}));
        auto leftBC = new BoundaryCondition(Dirichlet, new map<DOFType, double>(
                                                                        {{Temperature, 0}}));
        auto dummyBCMap = new map<Position, BoundaryCondition*>();
        dummyBCMap->insert(pair<Position, BoundaryCondition*>(Position::Left, leftBC));
        dummyBCMap->insert(pair<Position, BoundaryCondition*>(Position::Right, rightBC));
        dummyBCMap->insert(pair<Position, BoundaryCondition*>(Position::Top, topBC));
        dummyBCMap->insert(pair<Position, BoundaryCondition*>(Position::Bottom, bottomBC));
        
        return new DomainBoundaryConditions(dummyBCMap);*/
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