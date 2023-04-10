//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "MeshSpecs.h"
#include "../Discretization/Mesh/Mesh.h"
#include "../Discretization/Mesh/Mesh1D.h"
#include "../Discretization/Mesh/Mesh2D.h"
#include "../Discretization/Mesh/Mesh3D.h"
#include "../LinearAlgebra/Array.h"
#include "../LinearAlgebra/Transformations.h"
#include "../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"
#include "NodeFactory.h"
namespace StructuredMeshGenerator {
    
    class MeshFactory {
    public:
        MeshFactory(MeshSpecs *meshSpecs);
        Mesh *mesh;

        
    private:
        MeshSpecs *_meshSpecs;

        Mesh* initiateRegularMesh();
        Mesh* initiateGhostMesh(Mesh* targetMesh, map<Direction, unsigned> depth) const;
        void assignCoordinates();
        void assign1DCoordinates() const;
        void assign2DCoordinates() const;
        void assign3DCoordinates() const;
        void calculateMeshMetrics();
        SpaceEntityType calculateSpaceEntityType();
    };

};