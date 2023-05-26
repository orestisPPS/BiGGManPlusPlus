//
// Created by hal9000 on 3/13/23.
//

#include "StStFDTest.h"
#include "../../Discretization/Mesh/GhostPseudoMesh/GhostPseudoMesh.h"
#include "../../Utility/Exporters/Exporters.h"



namespace NumericalAnalysis {
    
    StStFDTest::StStFDTest() {
        map<Direction, unsigned> numberOfNodes;

        numberOfNodes[Direction::One] = 20;
        numberOfNodes[Direction::Two] = 20;
        //auto specs = new StructuredMeshGenerator::MeshSpecs(numberOfNodes, 0.25, 0.25, 0, 0, 0);
        auto specs = new MeshSpecs(numberOfNodes, 1, 1, 0, 0, 0);
        auto meshFactory = new MeshFactory(specs);
        meshFactory->domainBoundaryFactory->parallelogram(numberOfNodes, 19, 19);
        meshFactory->buildMesh(2);
        auto filenameParaview = "firstMesh.vtk";
        auto path = "/home/hal9000/code/BiGGMan++/Testing/";
        auto mesh = meshFactory->mesh;
        mesh->storeMeshInVTKFile(path, filenameParaview);
    }

    DomainBoundaryConditions* StStFDTest::createBC(Mesh *mesh) {
        
        /*auto boundaryFactory = new DomainBoundaryFactory(mesh);
        return boundaryFactory->parallelogram(101, 101, 100, 100);*/

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