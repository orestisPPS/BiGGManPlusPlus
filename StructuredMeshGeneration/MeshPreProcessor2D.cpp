//
// Created by hal9000 on 12/17/22.
//

#include "MeshPreProcessor.h"

namespace StructuredMeshGenerator{
    
    MeshPreProcessor :: MeshPreProcessor(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space){
        mesh = InitiateMesh(meshSpecs, space);
        AssignSpatialProperties(meshSpecs, space);
        AssignCoordinates(meshSpecs, space);
       // CalculateMeshMetrics();
    }

    Mesh* MeshPreProcessor::InitiateMesh(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space) {
        auto nodeFactory = NodeFactory(meshSpecs.nodesPerDirection, (PhysicalSpaceEntity &) space);
        return new Mesh(nodeFactory.nodesMatrix, &space);
    }
    
    void MeshPreProcessor::AssignSpatialProperties(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space) const {
        mesh->getSpatialProperties(meshSpecs.nodesPerDirection, space, meshSpecs.dimensions, meshSpecs.nodesPerDirection[One] * meshSpecs.nodesPerDirection[Two] * meshSpecs.nodesPerDirection[Three]);
    }
    
    void MeshPreProcessor::AssignCoordinates(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space) {
        if (space.type() == One_axis || space.type() == Two_axis || space.type() == Three_axis) {
            Assign1DCoordinates(meshSpecs, space);
        }
        else if (space.type() == OneTwo_plane || space.type() == TwoThree_plane || space.type() == OneThree_plane) {
            Assign2DCoordinates(meshSpecs, space);
        }
        else if (space.type() == OneTwoThree_volume) {
            Assign3DCoordinates(meshSpecs, space);
        }
    }
    
    void MeshPreProcessor::Assign1DCoordinates(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space) const {
        auto direction = Direction::One;
        if (space.type() == Two_axis)
            direction = Direction::Two;
        else if (space.type() == Three_axis)
            direction = Direction::Three;

        for (unsigned i = 0; i < mesh->numberOfNodesPerDirection.at(direction); ++i) {
            mesh->node(i)->setPositionVector(Natural);
            mesh->node(i)->setPositionVector({static_cast<double>(i)}, Parametric);
            mesh->node(i)->setPositionVector({static_cast<double>(i) * meshSpecs.templateStepOne}, Template);
        }
    }
    
    void MeshPreProcessor::Assign2DCoordinates(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space) const {
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
    
    void MeshPreProcessor::Assign3DCoordinates(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space) const {
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