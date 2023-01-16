//
// Created by hal9000 on 12/17/22.
//

#include "MeshPreProcessor.h"

namespace StructuredMeshGenerator{
    
    MeshPreProcessor :: MeshPreProcessor(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space){
        InitiateMesh(meshSpecs, space);
        AssignSpatialProperties(meshSpecs, space);
        AssignCoordinates();
       // CalculateMeshMetrics();
    }

    void MeshPreProcessor::InitiateMesh(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space) {
        auto nodeFactory = NodeFactory(meshSpecs.nodesPerDirection, (PhysicalSpaceEntity &) space.type());
        mesh = new Mesh(nodeFactory.nodesMatrix, &space);
    }
    
    void MeshPreProcessor::AssignSpatialProperties(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space) {
        
    }
    
    void MeshPreProcessor::AssignCoordinates() {
        for (auto k = 0; k < mesh->numberOfNodesPerDirection.at(Direction::Three); ++k) {
            for (auto j = 0; j < mesh->numberOfNodesPerDirection.at(Direction::Two); ++j) {
                for (auto i = 0; i < mesh->numberOfNodesPerDirection.at(Direction::One); ++i) {
                    switch (spaceCharacteristics->physicalSpace) {
                        case One_axis:
                            mesh->node(i)->addCoordinate(PositioningInSpace::Natural, Direction::One);
                            break;
                        case Two_axis:
                            mesh->node(i)->addCoordinate(PositioningInSpace::Natural, Direction::Two);
                            break;
                        case Three_axis:
                            mesh->node(i)->addCoordinate(PositioningInSpace::Natural, Direction::Three);
                            break;
                        case OneTwo_plane:
                            mesh->node(i, j)->addCoordinate(PositioningInSpace::Natural, Direction::One);
                            mesh->node(i, j)->addCoordinate(PositioningInSpace::Natural, Direction::Two);
                            break;
                        case OneThree_plane:
                            mesh->node(i, k)->addCoordinate(PositioningInSpace::Natural, Direction::One);
                            mesh->node(i, k)->addCoordinate(PositioningInSpace::Natural, Direction::Three);
                            break;
                        case TwoThree_plane:
                            mesh->node(j, k)->addCoordinate(PositioningInSpace::Natural, Direction::Two);
                            mesh->node(j, k)->addCoordinate(PositioningInSpace::Natural, Direction::Three);
                            break;
                        case OneTwoThree_volume:
                            mesh->node(i, j, k)->addCoordinate(PositioningInSpace::Natural, Direction::One);
                            mesh->node(i, j, k)->addCoordinate(PositioningInSpace::Natural, Direction::Two);
                            mesh->node(i, j, k)->addCoordinate(PositioningInSpace::Natural, Direction::Three);
                            break;
                    }
                }
            }
        }
    }
            
    void MeshPreProcessor::Assign1DCoordinates(Direction direction1) {
        for (auto i = 0; i < mesh->numberOfNodesPerDirection.at(direction1); ++i) {
            mesh->node(i)->addCoordinate(PositioningInSpace::Natural, direction1);
            mesh->node(i)->addCoordinate(PositioningInSpace::Parametric, direction1, i);
        }
        
    }
    
    void MeshPreProcessor::Assign2DCoordinates() {
        throw runtime_error("Not Implemented!");
    }
    
    void MeshPreProcessor::Assign3DCoordinates() {
        throw runtime_error("Not Implemented!");
    }
    
    void MeshPreProcessor::CalculateMeshMetrics() {
        throw runtime_error("Not Implemented!");
    }
}// StructuredMeshGenerator