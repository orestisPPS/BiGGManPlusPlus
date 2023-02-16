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
        static Mesh* initiateMesh(MeshSpecs &meshSpecs);
        void assignSpatialProperties(MeshSpecs &meshSpecs) const;
        void assignCoordinates(MeshSpecs &meshSpecs);
        void assign1DCoordinates(MeshSpecs &meshSpecs) const;
        void assign2DCoordinates(MeshSpecs &meshSpecs) const;
        void assign3DCoordinates(MeshSpecs &meshSpecs) const;
        void CalculateMeshMetrics();
        static SpaceEntityType calculateSpaceEntityType(MeshSpecs &meshSpecs) ;
    };

};