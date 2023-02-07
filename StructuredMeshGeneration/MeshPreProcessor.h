//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "MeshSpecs.h"
#include "../Discretization/Mesh/Mesh.h"
#include "../LinearAlgebra/Array.h"
#include "../LinearAlgebra/Transformations.h"
#include "../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"
#include "NodeFactory.h"
namespace StructuredMeshGenerator {
    
    class MeshPreProcessor {
    public:
        MeshPreProcessor(MeshSpecs &meshSpecs);
        Mesh *mesh;
        
    private:
        static Mesh* InitiateMesh(MeshSpecs &meshSpecs);
        void AssignSpatialProperties(MeshSpecs &meshSpecs) const;
        void AssignCoordinates(MeshSpecs &meshSpecs);
        void Assign1DCoordinates(MeshSpecs &meshSpecs) const;
        void Assign2DCoordinates(MeshSpecs &meshSpecs) const;
        void Assign3DCoordinates(MeshSpecs &meshSpecs) const;
        void CalculateMeshMetrics();
    };

};