//
// Created by hal9000 on 12/17/22.
//

#include "MeshPreProcessor.h"

namespace StructuredMeshGenerator{
    
    MeshPreProcessor :: MeshPreProcessor(MeshSpecs &meshSpecs) : meshSpecs(meshSpecs){
        spaceCharacteristics = new SpaceCharacteristics(meshSpecs.nodesPerDirection, CoordinateSystem::Parametric_Cartesian);
        InitiateMesh();
        mesh->spaceCharacteristics = spaceCharacteristics;
        AssignCoordinatesToNodes();
       // CalculateMeshMetrics();
    }

    void MeshPreProcessor::InitiateMesh() {
        auto nodeFactory = NodeFactory(meshSpecs.nodesPerDirection, spaceCharacteristics);
        mesh = new Mesh(nodeFactory.nodesMatrix);
    }
    
    void MeshPreProcessor::AssignCoordinatesToNodes() {
        switch (mesh->MeshDimensions()) {
            case 1:
                Assign1DCoordinates();
                break;
            case 2:
                Assign2DCoordinates();
                break;
            case 3:
                Assign3DCoordinates();
                break;
            default:
                throw runtime_error("Mesh dimensions should be 1, 2 or 3");
        }
        
    }
        
    void MeshPreProcessor::Assign1DCoordinates() {
        throw runtime_error("Not Implemented!");
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