//
// Created by hal9000 on 2/18/23.
//

#include "DOFInitializer.h"

namespace Analysis {
    
    DOFInitializer::DOFInitializer(Mesh *mesh,
                                   DomainBoundaryConditions *domainBoundaryConditions,
                                   struct Field_DOFType degreesOfFreedom){
        freeDegreesOfFreedom = new list<DegreeOfFreedom*>();
        boundedDegreesOfFreedom = new list<DegreeOfFreedom*>();
        fluxDegreesOfFreedom = new list<DegreeOfFreedom*>();
        
        auto dofStruct = new struct DisplacementVectorField3D_DOFType();
        auto lol = dofStruct->DegreesOfFreedom->at(0);
        dofStruct->deallocate();
        
        auto lel = new list<DOFType*> {lol};
        addDOFToInternalNodes(mesh, lel);
        
        void lil(Mesh *mesh, list<DOFType*> degreesOfFreedom );
    }
    
    void DOFInitializer::addDOFToInternalNodes(Mesh *mesh, list<DOFType*>* degreesOfFreedom ){
        for (int i = 0; i < degreesOfFreedom.size(); ++i) {
            for (int k = 0; k < mesh->numberOfNodesPerDirection[PositioningInSpace::Three]; ++k) {
                for (int j = 0; j < mesh->numberOfNodesPerDirection[PositioningInSpace::Two]; ++j) {
                    for (int l = 0; l < mesh->numberOfNodesPerDirection[PositioningInSpace::One]; ++l) {
                        switch (mesh->space()) {
                            case Axis:
                                break;  
                                
                            case Plane:
                                break;
                            
                            case Volume:
                                break;
                            
                        }
                    }
                }
                
            }
        }
    }

    
    
} // DOFInitializer