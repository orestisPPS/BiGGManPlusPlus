//
// Created by hal9000 on 2/18/23.
//

#include "DOFInitializer.h"

namespace Analysis {
    
    DOFInitializer::DOFInitializer(Mesh* mesh,DomainBoundaryConditions* domainBoundaryConditions, Field_DOFType* degreesOfFreedom) {
        freeDegreesOfFreedom = new list<DegreeOfFreedom*>();
        boundedDegreesOfFreedom = new list<DegreeOfFreedom*>();
        fluxDegreesOfFreedom = new list<DegreeOfFreedom*>();
        totalDegreesOfFreedom = new list<DegreeOfFreedom*>();
        initiateInternalNodeDOFs(mesh, degreesOfFreedom);
        initiateBoundaryNodeDOFs(mesh, degreesOfFreedom, domainBoundaryConditions);
    }
    
    void DOFInitializer::initiateInternalNodeDOFs(Mesh *mesh, Field_DOFType *degreesOfFreedom) const {
        //March through the mesh nodes
        for (int k = 1; k < mesh->numberOfNodesPerDirection[PositioningInSpace::Three] - 1; k++)
            for (int j = 1; j < mesh->numberOfNodesPerDirection[PositioningInSpace::Two] - 1; j++)
                for (int i = 1; i < mesh->numberOfNodesPerDirection[PositioningInSpace::One] - 1; i++) {
                    //March through the degrees of freedom
                    for (int l = 0; l < degreesOfFreedom->DegreesOfFreedom->size(); ++l) {
                        unsigned * node;
                        switch (mesh->space()) {
                            case Volume:
                                node = mesh->node(i, j, k)->id.global;
                                break;
                            case Plane:
                                node = mesh->node(i, j)->id.global;
                                break;
                            case Axis:
                                node = mesh->node(i)->id.global;
                                break;
                            default:
                                throw runtime_error("Mesh is F'd up");
                        }                          
                        auto dof = new DegreeOfFreedom(degreesOfFreedom->DegreesOfFreedom->at(l), node, false);
                        freeDegreesOfFreedom->push_back(dof);
                        totalDegreesOfFreedom->push_back(dof);
                    }
                }
    }

    void DOFInitializer::initiateBoundaryNodeDOFs(Mesh *mesh, Field_DOFType *degreesOfFreedom,
                                                  DomainBoundaryConditions *domainBoundaryConditions) {
        

    }
}