//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "MeshSpecs.h"
#include "../Discretization/Mesh/Mesh.h"
#include "../LinearAlgebra/Array.h"
#include "NodeFactory.h"
#include "../PositioningInSpace/SpaceCharacteristics.h"
namespace StructuredMeshGenerator {
    
    class MeshPreProcessor {
    public:
        MeshPreProcessor(MeshSpecs &meshSpecs);
        MeshSpecs &meshSpecs;
        SpaceCharacteristics *spaceCharacteristics;
        Mesh *mesh;
        
    private:
        void InitiateMesh();
        void AssignCoordinatesToNodes();
        void Assign1DCoordinates();
        void Assign2DCoordinates();
        void Assign3DCoordinates();
        void CalculateMeshMetrics();

    };

};