//
// Created by hal9000 on 3/13/23.
//

#include "StStFDTest.h"


namespace NumericalAnalysis {
    
    StStFDTest::StStFDTest() {
        auto mesh = createMesh();
        auto pde = createPDE();
        auto bcs = createBC();
        auto problem = new SteadyStateMathematicalProblem(pde, bcs, createDOF());
        auto schemeSpecs = createSchemeSpecs();
        auto analysis = new SteadyStateFiniteDifferenceAnalysis(problem, mesh, schemeSpecs);
    }

    Mesh* StStFDTest::createMesh() {
        map<Direction, unsigned> numberOfNodes;
        numberOfNodes[Direction::One] = 5;
        numberOfNodes[Direction::Two] = 5;
        auto specs = StructuredMeshGenerator::MeshSpecs(numberOfNodes, 1, 1, 0, 0, 0);
        auto space = (PositioningInSpace::Plane);
        auto mesh = StructuredMeshGenerator::MeshPreProcessor(specs).mesh;
        return mesh;            
    }
    
    PartialDifferentialEquation* StStFDTest::createPDE() {
        return new PartialDifferentialEquation(Laplace);
    }
    
    DomainBoundaryConditions* StStFDTest::createBC() {
        auto dirichletLeft = new BoundaryCondition(function<double (vector<double>*)> ([](vector<double>* x) {return 0;}));
        auto dirichletRight = new BoundaryCondition(function<double (vector<double>*)> ([](vector<double>* x) {return 0;}));
        auto dirichletTop = new BoundaryCondition(function<double (vector<double>*)> ([](vector<double>* x) {return 0;}));
        auto dirichletBottom = new BoundaryCondition(function<double (vector<double>*)> ([](vector<double>* x) {return 0;}));
        auto bcs = new DomainBoundaryConditions(Plane);
        bcs->AddDirichletBoundaryConditions(Left, new list<BoundaryCondition*> {dirichletLeft});
        bcs->AddDirichletBoundaryConditions(Right, new list<BoundaryCondition*> {dirichletRight});
        bcs->AddDirichletBoundaryConditions(Top, new list<BoundaryCondition*> {dirichletTop});
        bcs->AddDirichletBoundaryConditions(Bottom, new list<BoundaryCondition*> {dirichletBottom});
        return bcs;
    }
    
    Field_DOFType* StStFDTest::createDOF() {
        return new TemperatureScalar_DOFType();
    }
    
    FDSchemeSpecs* StStFDTest::createSchemeSpecs() {
        auto dummyScheme = map<Direction, tuple<FiniteDifferenceSchemeType,int>>();
        auto space = PositioningInSpace::Plane;
        return new FDSchemeSpecs(dummyScheme, space);
    }
    
    
} // NumericalAnalysis