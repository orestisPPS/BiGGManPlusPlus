//
// Created by hal9000 on 12/17/22.
//

#include "MeshFactory.h"
#include "../LinearAlgebra/Solvers/Iterative/StationaryIterative/SORSolver.h"


namespace StructuredMeshGenerator{
    
    MeshFactory :: MeshFactory(MeshSpecs *meshSpecs) : _meshSpecs(meshSpecs) {
        mesh = _initiateRegularMesh();
        _assignCoordinates();
        mesh->specs = _meshSpecs;
        mesh->calculateMeshMetrics(Template, true);
        _calculatePDEPropertiesFromMetrics();
        domainBoundaryFactory = new DomainBoundaryFactory(mesh);
        
    }
    
    void MeshFactory::buildMesh(unsigned short schemeOrder) const {
        
        auto start = chrono::steady_clock::now();
        
        auto pdeProperties = new SecondOrderLinearPDEProperties(2, false, LocallyAnisotropic);
        pdeProperties->setLocallyAnisotropicProperties(pdePropertiesFromMetrics);

        auto pde = new PartialDifferentialEquation(pdeProperties, Laplace);
        
        auto specs = new FDSchemeSpecs(schemeOrder, schemeOrder, mesh->directions());
        
        DomainBoundaryConditions* boundaryConditions = nullptr;
        if (_boundaryFactoryInitialized) {
            boundaryConditions = domainBoundaryFactory->getDomainBoundaryConditions();
        }
        else
            throw invalid_argument("Boundary conditions not initialized");
        
        Field_DOFType* dofTypes = nullptr;
        switch (mesh->dimensions()) {
            case 1:
                dofTypes = new nodalPositionVectorField1D_DOFType();
                break;
            case 2:
                dofTypes = new nodalPositionVectorField2D_DOFType();
                break;
            case 3:
                dofTypes = new nodalPositionVectorField3D_DOFType();
                break;
        }
        
        auto problem = new SteadyStateMathematicalProblem(pde, boundaryConditions, dofTypes);

        //auto solver = new SolverLUP(1E-20, true);
        //auto solver  = new JacobiSolver(true, VectorNormType::L1, 1E-8, 1E4, true);
        //auto solver  = new GaussSeidelSolver(true, VectorNormType::LInf, 1E-9);
        auto solver = new SORSolver(1.8, true, VectorNormType::LInf, 1E-9);
        

        auto analysis =
                new SteadyStateFiniteDifferenceAnalysis(problem, mesh, solver, specs, Parametric);

        analysis->solve();
        
        analysis->applySolutionToDegreesOfFreedom();
        
        for (auto &node : *mesh->totalNodesVector){
            auto coords = new vector<double>();
            for (auto &dof : *node->degreesOfFreedom){
                coords->push_back(dof->value());
            }
/*            for (auto &dof : *node->degreesOfFreedom){
                delete dof;
                dof = nullptr;
            }*/
            node->degreesOfFreedom->clear();
            node->coordinates.addPositionVector(coords);
        }
        
        delete specs;
        //delete problem;
        delete solver;
        delete analysis->linearSystem;
        //delete analysis->degreesOfFreedom;
        dofTypes->deallocate();
        delete dofTypes;


        auto end = chrono::steady_clock::now();
        cout<< "Mesh Built in " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
    }
    
    Mesh* MeshFactory::_initiateRegularMesh() {
        auto nodesPerDirection = _meshSpecs->nodesPerDirection;
        auto nodeFactory = NodeFactory(_meshSpecs->nodesPerDirection);
        auto space = _calculateSpaceEntityType();
        switch (space) {
            case Axis:
                return new Mesh1D(nodeFactory.nodesMatrix);
            case Plane:
                return new Mesh2D(nodeFactory.nodesMatrix);
            case Volume:
                return new Mesh3D(nodeFactory.nodesMatrix);
            default:
                throw runtime_error("Invalid space type");
        }
    }
    
    void MeshFactory::_assignCoordinates() {
        switch (_calculateSpaceEntityType()) {
            case Axis:
                _assign1DCoordinates();
                break;
            case Plane:
                _assign2DCoordinates();
                break;
            case Volume:
                _assign3DCoordinates();
                break;
            default:
                throw runtime_error("Invalid space type");
        }
        
        auto space = _calculateSpaceEntityType();
        if (space == Axis) {
            _assign1DCoordinates();
        } else if (space == Plane) {
            _assign2DCoordinates();
        } else {
            _assign3DCoordinates();
        }
    }
    
    void MeshFactory::_assign1DCoordinates() const {
        for (unsigned i = 0; i < mesh->nodesPerDirection.at(One); ++i) {
            //mesh->node(i)->coordinates.addPositionVector(Natural);
            mesh->node(i)->coordinates.setPositionVector(
                    new vector<double>{static_cast<double>(i)}, Parametric);
            mesh->node(i)->coordinates.setPositionVector(
                    new vector<double>{static_cast<double>(i) * _meshSpecs->templateStepOne}, Template);
        }
    }
    
    void MeshFactory::_assign2DCoordinates() const {
        for (unsigned j = 0; j < mesh->nodesPerDirection.at(Two); ++j) {
            for (unsigned i = 0; i < mesh->nodesPerDirection.at(One); ++i) {
                
                // Parametric coordinates
                mesh->node(i, j)->coordinates.addPositionVector(
                        new vector<double>{static_cast<double>(i), static_cast<double>(j)}, Parametric);
                // Template coordinates
                vector<double> templateCoord = {static_cast<double>(i) * _meshSpecs->templateStepOne,
                                                static_cast<double>(j) * _meshSpecs->templateStepTwo};
                // Rotate 
                Transformations::rotate(templateCoord, _meshSpecs->templateRotAngleOne);
                // Shear
                Transformations::shear(templateCoord, _meshSpecs->templateShearOne,_meshSpecs->templateShearTwo);

                mesh->node(i, j)->coordinates.addPositionVector(new vector<double>(templateCoord), Template);
            }
        }
    }
    
    void MeshFactory::_assign3DCoordinates() const {
        for (unsigned k = 0; k < mesh->nodesPerDirection.at(Three); ++k) {
            for (unsigned j = 0; j < mesh->nodesPerDirection.at(Two); ++j) {
                for (unsigned i = 0; i < mesh->nodesPerDirection.at(One); ++i) {
                    // Natural coordinates
                    //mesh->node(i, j, k)->coordinates.addPositionVector(Natural);
                    // Parametric coordinates
                    mesh->node(i, j, k)->coordinates.addPositionVector(
                            new vector<double>{static_cast<double>(i), static_cast<double>(j), static_cast<double>(k)}, Parametric);
                    // Template coordinates
                    vector<double> templateCoord = {static_cast<double>(i) * _meshSpecs->templateStepOne,
                                                    static_cast<double>(j) * _meshSpecs->templateStepTwo,
                                                    static_cast<double>(k) * _meshSpecs->templateStepThree};
                    // Rotate 
                    Transformations::rotate(templateCoord, _meshSpecs->templateRotAngleOne);
                    // Shear
                    Transformations::shear(templateCoord, _meshSpecs->templateShearOne,_meshSpecs->templateShearTwo);
                    
                    mesh->node(i, j, k)->coordinates.addPositionVector(new vector<double>(templateCoord), Template);
                }
            }
        }
    }
    
    SpaceEntityType MeshFactory::_calculateSpaceEntityType(){
        auto space = NullSpace;
        if (_meshSpecs->nodesPerDirection[Two]== 1 && _meshSpecs->nodesPerDirection[Three] == 1){
            space = Axis;
        } else if (_meshSpecs->nodesPerDirection[Three] == 1){
            space = Plane;
        } else {
            space = Volume;
        }
        return space;
    }

    void MeshFactory::_calculatePDEPropertiesFromMetrics() {
        pdePropertiesFromMetrics = new map<unsigned, FieldProperties>();
        for (auto &node : *mesh->totalNodesVector) {
            auto nodeFieldProperties = FieldProperties();
            auto loliti = *mesh->metrics->at(*node->id.global)->contravariantTensor;
            nodeFieldProperties.secondOrderCoefficients = mesh->metrics->at(*node->id.global)->contravariantTensor;
            nodeFieldProperties.firstOrderCoefficients = new vector<double>(2, 0);
            nodeFieldProperties.zerothOrderCoefficient = new double(0);
            nodeFieldProperties.sourceTerm = new double(0);
            pdePropertiesFromMetrics->insert(pair<unsigned, FieldProperties>(*node->id.global, nodeFieldProperties));
        }
/*        for (auto &metrics : *mesh->metrics) {
            delete metrics.second;
        }
        delete mesh->metrics;*/
    }



}// StructuredMeshGenerator

