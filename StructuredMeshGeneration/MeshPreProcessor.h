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
        MeshPreProcessor(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space);
        Mesh *mesh;
        
    private:
        void InitiateMesh(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space);
        void AssignSpatialProperties(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space) const;
        void AssignCoordinates(PhysicalSpaceEntity &space);
        void Assign1DCoordinates(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space);
        void Assign2DCoordinates(MeshSpecs &meshSpecs, PhysicalSpaceEntity &space);
        void Assign3DCoordinates(Direction direction1, Direction direction2, Direction direction3);
        void CalculateMeshMetrics();
    };

};