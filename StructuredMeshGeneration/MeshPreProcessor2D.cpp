//
// Created by hal9000 on 12/17/22.
//

#include "MeshPreProcessor.h"

namespace StructuredMeshGenerator{
    
    MeshPreProcessor :: MeshPreProcessor(MeshSpecs &meshSpecs){
        mesh = InitiateMesh(meshSpecs);
        AssignSpatialProperties(meshSpecs);
        AssignCoordinates(meshSpecs);
       // CalculateMeshMetrics();
    }

    Mesh* MeshPreProcessor::InitiateMesh(MeshSpecs &meshSpecs) {
        auto nodeFactory = NodeFactory(meshSpecs.nodesPerDirection);
        return new Mesh(nodeFactory.nodesMatrix);
    }
    
    void MeshPreProcessor::AssignSpatialProperties(MeshSpecs &meshSpecs) const {
        mesh->getSpatialProperties(meshSpecs.nodesPerDirection, meshSpecs.dimensions, meshSpecs.nodesPerDirection[One] * meshSpecs.nodesPerDirection[Two] * meshSpecs.nodesPerDirection[Three]);
    }
    
    void MeshPreProcessor::AssignCoordinates(MeshSpecs &meshSpecs) {
        if (mesh->space() == Axis)
            Assign1DCoordinates(meshSpecs);
        else if (mesh->space() == Plane)
            Assign2DCoordinates(meshSpecs);
        else if (mesh->space() == Volume)
            Assign3DCoordinates(meshSpecs);
    }
    
    void MeshPreProcessor::Assign1DCoordinates(MeshSpecs &meshSpecs) const {
        for (unsigned i = 0; i < mesh->numberOfNodesPerDirection.at(One); ++i) {
            mesh->node(i)->coordinates.addPositionVector
            mesh->node(i)->coordinates.setPositionVector({static_cast<double>(i)}, Parametric);
            mesh->node(i)->coordinates.setPositionVector({static_cast<double>(i) * meshSpecs.templateStepOne}, Template);
        }
    }
    
    void MeshPreProcessor::Assign2DCoordinates(MeshSpecs &meshSpecs) const {
        auto direction1 = Direction::One;
        auto direction2 = Direction::Two;
        if (space.type() == OneTwo_plane) {
            direction1 = Direction::One;
            direction2 = Direction::Two;
        } else if (space.type() == OneThree_plane) {
            direction1 = Direction::One;
            direction2 = Direction::Three;
        } else if (space.type() == TwoThree_plane) {
            direction1 = Direction::Two;
            direction2 = Direction::Three;
        }
        for (unsigned j = 0; j < mesh->numberOfNodesPerDirection.at(direction2); ++j) {
            for (unsigned i = 0; i < mesh->numberOfNodesPerDirection.at(direction1); ++i) {
                
                // Natural coordinates
                mesh->node(i, j)->setPositionVector(Natural);
                // Parametric coordinates
                mesh->node(i, j)->setPositionVector({static_cast<double>(i), static_cast<double>(j)}, Parametric);
                // Template coordinates
                vector<double> templateCoord = {static_cast<double>(i) * meshSpecs.templateStepOne,
                                                static_cast<double>(j) * meshSpecs.templateStepTwo};
                // Rotate 
                templateCoord = Transformations::rotatePlane(templateCoord, meshSpecs.templateRotAngleOne, space.type());
                // Shear
                templateCoord = Transformations::shearInPlane(templateCoord, meshSpecs.templateShearOne,meshSpecs.templateShearTwo, space.type());
                
                mesh->node(i, j)->setPositionVector(templateCoord, Template);
            }   
        }
    }
    
    void MeshPreProcessor::Assign3DCoordinates(MeshSpecs &meshSpecs) const {
        for (unsigned k = 0; k < mesh->numberOfNodesPerDirection.at(Three); ++k) {
            for (unsigned j = 0; j < mesh->numberOfNodesPerDirection.at(Two); ++j) {
                for (unsigned i = 0; i < mesh->numberOfNodesPerDirection.at(One); ++i) {
                    // Natural coordinates
                    mesh->node(i, j, k)->setPositionVector(Natural);
                    // Parametric coordinates
                    mesh->node(i, j, k)->setPositionVector({static_cast<double>(i), static_cast<double>(j), static_cast<double>(k)}, Parametric);
                    // Template coordinates
                    vector<double> templateCoord = {static_cast<double>(i) * meshSpecs.templateStepOne,
                                                    static_cast<double>(j) * meshSpecs.templateStepTwo,
                                                    static_cast<double>(k) * meshSpecs.templateStepThree};
                    // Rotate 
                    templateCoord = Transformations::rotatePlane(templateCoord, meshSpecs.templateRotAngleOne, space.type());
                    // Shear
                    templateCoord = Transformations::shearInPlane(templateCoord, meshSpecs.templateShearOne,meshSpecs.templateShearTwo, space.type());
                }
            }
        }
    }
    
    void MeshPreProcessor::CalculateMeshMetrics() {
        throw runtime_error("Not Implemented!");
    }
}// StructuredMeshGenerator