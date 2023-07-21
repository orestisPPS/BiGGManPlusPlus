//
// Created by hal9000 on 12/17/22.
//

#include "MeshFactory.h"

#include <utility>
#include "../LinearAlgebra/Solvers/Iterative/StationaryIterative/SORSolver.h"


namespace StructuredMeshGenerator{
    
    MeshFactory :: MeshFactory(shared_ptr<MeshSpecs> meshSpecs) : _meshSpecs(std::move(meshSpecs)) {
        mesh = _initiateRegularMesh();
        _assignCoordinates();
        mesh->specs = _meshSpecs;
        mesh->calculateMeshMetrics(Template, true);
        _calculatePDEPropertiesFromMetrics();
    }
    
    void MeshFactory::buildMesh(unsigned short schemeOrder, shared_ptr<DomainBoundaryConditions> boundaryConditions) const {
        
        auto start = chrono::steady_clock::now();
        
        auto pdeProperties = make_shared<SecondOrderLinearPDEProperties>(2, false, LocallyAnisotropic);
        
        pdeProperties->setLocallyAnisotropicProperties(pdePropertiesFromMetrics);

        auto pde = make_shared<PartialDifferentialEquation>(pdeProperties, Laplace);
        
        auto specs = make_shared<FDSchemeSpecs>(schemeOrder, schemeOrder, mesh->directions());
        

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
        
        auto problem = make_shared<SteadyStateMathematicalProblem>(pde, boundaryConditions, dofTypes);

        //auto solver = make_shared<SolverLUP>(1E-20, true);
        //auto solver  = new JacobiSolver(true, VectorNormType::L1, 1E-8, 1E4, true);
        //auto solver  = new GaussSeidelSolver(true, VectorNormType::LInf, 1E-9);
        auto solver = make_shared<SORSolver>(1.7, vTechKickInYoo, VectorNormType::LInf, 1E-9);
        

        auto analysis = make_shared<SteadyStateFiniteDifferenceAnalysis>(problem, mesh, solver, specs, Template);
        

         analysis->solve();
        
        analysis->applySolutionToDegreesOfFreedom();
        
        for (auto &node : *mesh->totalNodesVector){
            auto coords = make_shared<vector<double>>();
            for (auto &dof : *node->degreesOfFreedom){
                coords->push_back(dof->value());
            }
            node->degreesOfFreedom->clear();
            node->coordinates.addPositionVector(coords);
        }
        //delete analysis->degreesOfFreedom;
        dofTypes->deallocate();
        delete dofTypes;
        

        auto end = chrono::steady_clock::now();
        cout<< "Mesh Built in " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
    }
    
    shared_ptr<Mesh> MeshFactory::_initiateRegularMesh() {
        auto nodesPerDirection = _meshSpecs->nodesPerDirection;
        auto nodeFactory = NodeFactory(_meshSpecs->nodesPerDirection);
        auto space = _calculateSpaceEntityType();
        switch (space) {
            case Axis:
                return make_shared<Mesh1D>(nodeFactory.nodesMatrix);
            case Plane:
                return make_shared<Mesh2D>(nodeFactory.nodesMatrix);
            case Volume:
                return make_shared<Mesh3D>(nodeFactory.nodesMatrix);
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
            auto coords = {static_cast<double>(i)};
            mesh->node(i)->coordinates.setPositionVector(
                    make_shared<vector<double>>(coords), Template);
            coords = {static_cast<double>(i) * _meshSpecs->templateStepOne};
            mesh->node(i)->coordinates.setPositionVector(
                    make_shared<vector<double>>(coords), Parametric);   
        }
    }
    
    void MeshFactory::_assign2DCoordinates() const {
        for (unsigned j = 0; j < mesh->nodesPerDirection.at(Two); ++j) {
            for (unsigned i = 0; i < mesh->nodesPerDirection.at(One); ++i) {
                
                // Parametric coordinates
                vector<double> parametricCoord = {static_cast<double>(i), static_cast<double>(j)};
                mesh->node(i, j)->coordinates.addPositionVector(make_shared<vector<double>>(parametricCoord), Parametric);
                // Template coordinates
                vector<double> templateCoord = {static_cast<double>(i) * _meshSpecs->templateStepOne,
                                                static_cast<double>(j) * _meshSpecs->templateStepTwo};
                // Rotate 
                Transformations::rotate(templateCoord, _meshSpecs->templateRotAngleOne);
                // Shear
                Transformations::shear(templateCoord, _meshSpecs->templateShearOne,_meshSpecs->templateShearTwo);

                mesh->node(i, j)->coordinates.addPositionVector(make_shared<vector<double>>(templateCoord), Template);
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
                    vector<double> parametricCoord = {static_cast<double>(i), static_cast<double>(j), static_cast<double>(k)};
                    mesh->node(i, j, k)->coordinates.addPositionVector(make_shared<vector<double>>(parametricCoord), Parametric);
                    // Template coordinates
                    vector<double> templateCoord = {static_cast<double>(i) * _meshSpecs->templateStepOne,
                                                    static_cast<double>(j) * _meshSpecs->templateStepTwo,
                                                    static_cast<double>(k) * _meshSpecs->templateStepThree};
                    // Rotate 
                    //Transformations::rotate(templateCoord, _meshSpecs->templateRotAngleOne);
                    // Shear
                    //Transformations::shear(templateCoord, _meshSpecs->templateShearOne,_meshSpecs->templateShearTwo);
                    
                    mesh->node(i, j, k)->coordinates.addPositionVector(make_shared<vector<double>>(templateCoord), Template);
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
        pdePropertiesFromMetrics = make_shared<map<unsigned, SpaceFieldProperties>>();
        for (auto &node : *mesh->totalNodesVector) {
            auto nodeFieldProperties = SpaceFieldProperties();
            nodeFieldProperties.secondOrderCoefficients = mesh->metrics->at(*node->id.global)->contravariantTensor;
            auto firstDerivativeCoefficients = vector<double>{0, 0, 0};
            nodeFieldProperties.firstOrderCoefficients = make_shared<vector<double>>(firstDerivativeCoefficients);
            nodeFieldProperties.zerothOrderCoefficient = make_shared<double>(0);
            nodeFieldProperties.sourceTerm = make_shared<double>(0);
            pdePropertiesFromMetrics->insert(pair<unsigned, SpaceFieldProperties>(*node->id.global, nodeFieldProperties));
        }
/*        for (auto &metrics : *mesh->metrics) {
            delete metrics.second;
        }
        delete mesh->metrics;*/
    }



}// StructuredMeshGenerator

