//
// Created by hal9000 on 12/17/22.
//

#include "MeshPreProcessor.h"

namespace StructuredMeshGenerator{
    
    MeshPreProcessor :: MeshPreProcessor(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space){
        InitiateMesh(meshSpecs, space);
        AssignSpatialProperties(meshSpecs, space);
        AssignCoordinates(space);
       // CalculateMeshMetrics();
    }

    void MeshPreProcessor::InitiateMesh(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space) {
        auto nodeFactory = NodeFactory(meshSpecs.nodesPerDirection, (PhysicalSpaceEntity &) space.type());
        mesh = new Mesh(nodeFactory.nodesMatrix, &space);
    }
    
    void MeshPreProcessor::AssignSpatialProperties(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space) const {
        mesh -> numberOfNodesPerDirection = meshSpecs.nodesPerDirection;
        mesh -> space = &space;
    }
    
    void MeshPreProcessor::AssignCoordinates(PhysicalSpaceEntity &space) {
        for (auto k = 0; k < mesh->numberOfNodesPerDirection.at(Direction::Three); ++k) {
            for (auto j = 0; j < mesh->numberOfNodesPerDirection.at(Direction::Two); ++j) {
                for (auto i = 0; i < mesh->numberOfNodesPerDirection.at(Direction::One); ++i) {
                    switch (space.type()) {
                        case One_axis:
                            break;
                        case Two_axis:
                            break;
                        case Three_axis:
                            break;
                        case OneTwo_plane:
; 
                            break;
                        case OneThree_plane:
                            
                        case TwoThree_plane:
                 
                            break;
                        case OneTwoThree_volume:
                          
                            break;
                    }
                }
            }
        }
    }
            
    void MeshPreProcessor::Assign1DCoordinates(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space) {
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
    
    void MeshPreProcessor::Assign2DCoordinates(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space) {
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
                mesh->node(i, j)->setPositionVector(Natural);
                mesh->node(i, j)->setPositionVector({static_cast<double>(i), static_cast<double>(j)}, Parametric);
                
                auto templateCoord = {static_cast<double>(i) * meshSpecs.templateStepOne,
                                      static_cast<double>(j) * meshSpecs.templateStepTwo};

                templateCoord = Transformations::Rotate(templateCoord, meshSpecs.templateRotation);
                
                mesh->node(i, j)->setPositionVector({static_cast<double>(i) * meshSpecs.templateStepOne,
                                                                 static_cast<double>(j) * meshSpecs.templateStepTwo}, Template);
                
                mesh->node(i, j)->positionVector(Template);
            }
        }
    }
    }
    
/*    void MeshPreProcessor::Assign2DCoordinates() {
        throw runtime_error("Not Implemented!");
    }
    
    void MeshPreProcessor::Assign3DCoordinates() {
        throw runtime_error("Not Implemented!");
    }
    
    void MeshPreProcessor::CalculateMeshMetrics() {
        throw runtime_error("Not Implemented!");
    }*/
}// StructuredMeshGenerator