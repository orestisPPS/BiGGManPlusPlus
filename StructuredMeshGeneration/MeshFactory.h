//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "MeshSpecs.h"
#include "../Discretization/Mesh/Mesh.h"
#include "../Discretization/Mesh/Mesh1D.h"
#include "../Discretization/Mesh/Mesh2D.h"
#include "../Discretization/Mesh/Mesh3D.h"
#include "../LinearAlgebra/Array/Array.h"
#include "../LinearAlgebra/Operations/Transformations.h"
#include "../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"
#include "NodeFactory.h"
#include "MeshSpecs.h"
#include "DomainBoundaryFactory.h"
#include "../PartialDifferentialEquations/SecondOrderLinearPDEProperties.h"
#include "../Analysis/FiniteDifferenceAnalysis/SteadyStateFiniteDifferenceAnalysis.h"
#include "../LinearAlgebra/Solvers/Direct/SolverLUP.h"


using namespace PartialDifferentialEquations;
namespace StructuredMeshGenerator {
    
    class MeshFactory {
    public:
        explicit MeshFactory(MeshSpecs *meshSpecs);
        
        Mesh *mesh;

        map<unsigned, FieldProperties>* pdePropertiesFromMetrics;
        
        DomainBoundaryFactory* domainBoundaryFactory;
        
        void buildMesh(unsigned short schemeOrder);
        
    private:
        MeshSpecs *_meshSpecs;
        
        bool _boundaryFactoryInitialized;
        
        DomainBoundaryConditions* _boundaryConditions;

        Mesh* _initiateRegularMesh();
        
        void _assignCoordinates();
        
        void _assign1DCoordinates() const;
        
        void _assign2DCoordinates() const;
        
        void _assign3DCoordinates() const;
        
        SpaceEntityType _calculateSpaceEntityType();

        void _calculatePDEPropertiesFromMetrics();
        
        
    };

};