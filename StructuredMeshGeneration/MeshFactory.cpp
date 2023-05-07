//
// Created by hal9000 on 12/17/22.
//

#include "MeshFactory.h"

namespace StructuredMeshGenerator{
    
    MeshFactory :: MeshFactory(MeshSpecs *meshSpecs) : _meshSpecs(meshSpecs) {
        mesh = _initiateRegularMesh();
        _assignCoordinates();
        mesh->specs = _meshSpecs;
        cout<<mesh->dimensions()<<endl;
        mesh->calculateMeshMetrics(Template, true);
        _calculatePDEPropertiesFromMetrics();
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
            mesh->node(i)->coordinates.addPositionVector(Natural);
            mesh->node(i)->coordinates.setPositionVector(
                    new vector<double>{static_cast<double>(i)}, Parametric);
            mesh->node(i)->coordinates.setPositionVector(
                    new vector<double>{static_cast<double>(i) * _meshSpecs->templateStepOne}, Template);
        }
    }
    
    void MeshFactory::_assign2DCoordinates() const {
        for (unsigned j = 0; j < mesh->nodesPerDirection.at(Two); ++j) {
            for (unsigned i = 0; i < mesh->nodesPerDirection.at(One); ++i) {
                
                // Natural coordinates
                mesh->node(i, j)->coordinates.addPositionVector(Natural);
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

                mesh->node(i, j)->coordinates.setPositionVector(new vector<double>(templateCoord), Template);
            }   
        }
    }
    
    void MeshFactory::_assign3DCoordinates() const {
        for (unsigned k = 0; k < mesh->nodesPerDirection.at(Three); ++k) {
            for (unsigned j = 0; j < mesh->nodesPerDirection.at(Two); ++j) {
                for (unsigned i = 0; i < mesh->nodesPerDirection.at(One); ++i) {
                    // Natural coordinates
                    mesh->node(i, j, k)->coordinates.addPositionVector(Natural);
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
                    
                    mesh->node(i, j, k)->coordinates.setPositionVector(new vector<double>(templateCoord), Template);
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
            nodeFieldProperties.firstOrderCoefficients = new vector<double>{0, 0};
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

