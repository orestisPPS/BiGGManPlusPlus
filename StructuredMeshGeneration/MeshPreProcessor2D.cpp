//
// Created by hal9000 on 12/17/22.
//

#include "MeshPreProcessor2D.h"

namespace StructuredMeshGenerator{
    
    MeshPreProcessor2D :: MeshPreProcessor2D(MeshSpecs2D &meshSpecs) : meshSpecs(meshSpecs){
        InitiateMesh();
        //AssignCoordinatesToNodes();
       // CalculateMeshMetrics();
    }

    void MeshPreProcessor2D::InitiateMesh() {
        auto nodeFactory = NodeFactory(meshSpecs.nnx, meshSpecs.nny, 0);
        mesh = new Mesh(nodeFactory.nodesMatrix);
    }
    
    vo
    
    
} // StructuredMeshGenerator