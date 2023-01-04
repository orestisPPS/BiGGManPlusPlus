//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "MeshSpecs2D.h"
#include "../Discretization/Mesh/Mesh.h"
#include "../Primitives/Array.h"
#include "NodeFactory.h"
namespace StructuredMeshGenerator {
    
    class MeshPreProcessor2D {
    public:
        MeshPreProcessor2D(MeshSpecs2D &meshSpecs);
        Mesh *mesh;
        MeshSpecs2D &meshSpecs;
    private:
        void InitiateMesh();
        void AssignCoordinatesToNodes();
        void AssignCoordinatesTo1DNodes();
        void CalculateMeshMetrics();
    };

};