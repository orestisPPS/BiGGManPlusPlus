//
// Created by hal9000 on 12/17/22.
//

#include "MeshFactory.h"

namespace StructuredMeshGenerator{
    
    MeshFactory :: MeshFactory(MeshSpecs &meshSpecs) : _meshSpecs(meshSpecs) {
        mesh = initiateRegularMesh();
        assignCoordinates();
       // calculateMeshMetrics();
    }

    Mesh* MeshFactory::initiateRegularMesh() {
        auto nodesPerDirection = _meshSpecs.nodesPerDirection;
        auto nodeFactory = NodeFactory(_meshSpecs.nodesPerDirection);
        auto space = calculateSpaceEntityType();
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

    
    void MeshFactory::assignCoordinates() {
        switch (calculateSpaceEntityType()) {
            case Axis:
                assign1DCoordinates();
                break;
            case Plane:
                assign2DCoordinates();
                break;
            case Volume:
                assign3DCoordinates();
                break;
            default:
                throw runtime_error("Invalid space type");
        }
        
        auto space = calculateSpaceEntityType();
        if (space == Axis) {
            assign1DCoordinates();
        } else if (space == Plane) {
            assign2DCoordinates();
        } else {
            assign3DCoordinates();
        }
    }
    
    void MeshFactory::assign1DCoordinates() const {
        for (unsigned i = 0; i < mesh->numberOfNodesPerDirection.at(One); ++i) {
            mesh->node(i)->coordinates.addPositionVector(Natural);
            mesh->node(i)->coordinates.setPositionVector({static_cast<double>(i)}, Parametric);
            mesh->node(i)->coordinates.setPositionVector({static_cast<double>(i) * _meshSpecs.templateStepOne}, Template);
        }
    }
    
    void MeshFactory::assign2DCoordinates() const {
        for (unsigned j = 0; j < mesh->numberOfNodesPerDirection.at(Two); ++j) {
            for (unsigned i = 0; i < mesh->numberOfNodesPerDirection.at(One); ++i) {
                
                // Natural coordinates
                mesh->node(i, j)->coordinates.addPositionVector(Natural);
                // Parametric coordinates
                mesh->node(i, j)->coordinates.addPositionVector({static_cast<double>(i), static_cast<double>(j)}, Parametric);
                // Template coordinates
                vector<double> templateCoord = {static_cast<double>(i) * _meshSpecs.templateStepOne,
                                                static_cast<double>(j) * _meshSpecs.templateStepTwo};
                // Rotate 
                Transformations::rotate(templateCoord, _meshSpecs.templateRotAngleOne);
                // Shear
                Transformations::shear(templateCoord, _meshSpecs.templateShearOne,_meshSpecs.templateShearTwo);

                mesh->node(i, j)->coordinates.setPositionVector(templateCoord, Template);
            }   
        }
    }
    
    void MeshFactory::assign3DCoordinates() const {
        for (unsigned k = 0; k < mesh->numberOfNodesPerDirection.at(Three); ++k) {
            for (unsigned j = 0; j < mesh->numberOfNodesPerDirection.at(Two); ++j) {
                for (unsigned i = 0; i < mesh->numberOfNodesPerDirection.at(One); ++i) {
                    // Natural coordinates
                    mesh->node(i, j, k)->coordinates.addPositionVector(Natural);
                    // Parametric coordinates
                    mesh->node(i, j, k)->coordinates.addPositionVector(
                            {static_cast<double>(i), static_cast<double>(j), static_cast<double>(k)}, Parametric);
                    // Template coordinates
                    vector<double> templateCoord = {static_cast<double>(i) * _meshSpecs.templateStepOne,
                                                    static_cast<double>(j) * _meshSpecs.templateStepTwo,
                                                    static_cast<double>(k) * _meshSpecs.templateStepThree};
                    // Rotate 
                    Transformations::rotate(templateCoord, _meshSpecs.templateRotAngleOne);
                    // Shear
                    Transformations::shear(templateCoord, _meshSpecs.templateShearOne,_meshSpecs.templateShearTwo);
                }
            }
        }
    }
    
    SpaceEntityType MeshFactory::calculateSpaceEntityType(){
        auto space = NullSpace;
        if (_meshSpecs.nodesPerDirection[Two]== 1 && _meshSpecs.nodesPerDirection[Three] == 1){
            space = Axis;
        } else if (_meshSpecs.nodesPerDirection[Three] == 1){
            space = Plane;
        } else {
            space = Volume;
        }
        return space;
    }

}// StructuredMeshGenerator

