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
        mesh->getSpatialProperties(meshSpecs.nodesPerDirection,
                                   meshSpecs.dimensions,
                                   meshSpecs.nodesPerDirection[One] * meshSpecs.nodesPerDirection[Two] * meshSpecs.nodesPerDirection[Three],
                                   calculateSpaceEntityType(meshSpecs));
    }
    
    void MeshPreProcessor::AssignCoordinates(MeshSpecs &meshSpecs) {
        auto space = calculateSpaceEntityType(meshSpecs);
        if (space == Axis) {
            Assign1DCoordinates(meshSpecs);
        } else if (space == Plane) {
            Assign2DCoordinates(meshSpecs);
        } else if (space == PositioningInSpace::Volume) {
            Assign3DCoordinates(meshSpecs);
        }
        else {
            throw std::invalid_argument("Invalid space entity type");
        }
    }
    
    void MeshPreProcessor::Assign1DCoordinates(MeshSpecs &meshSpecs) const {
        for (unsigned i = 0; i < mesh->numberOfNodesPerDirection.at(One); ++i) {
            mesh->node(i)->coordinates.addPositionVector
            mesh->node(i)->coordinates.setPositionVector({static_cast<double>(i)}, Parametric);
            mesh->node(i)->coordinates.setPositionVector({static_cast<double>(i) * meshSpecs.templateStepOne}, Template);
        }
    }
    
    void MeshPreProcessor::Assign2DCoordinates(MeshSpecs &meshSpecs) const {
        for (unsigned j = 0; j < mesh->numberOfNodesPerDirection.at(Two); ++j) {
            for (unsigned i = 0; i < mesh->numberOfNodesPerDirection.at(One); ++i) {
                
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
    
    SpaceEntityType MeshPreProcessor::calculateSpaceEntityType(MeshSpecs &meshSpecs) const {
        auto space = NullSpace;
        if (meshSpecs.nodesPerDirection[Two]== 0 && meshSpecs.nodesPerDirection[Three] == 0){
            space = Axis;
        } else if (meshSpecs.nodesPerDirection[Three] == 0){
            space = Plane;
        } else {
            space = Volume;
        }
        return space;
        }

    }
    
    void MeshPreProcessor::CalculateMeshMetrics() {
        throw runtime_error("Not Implemented!");
    }
}// StructuredMeshGenerator