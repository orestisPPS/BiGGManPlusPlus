//
// Created by hal9000 on 4/10/23.
//

#include "GhostPseudoMesh.h"

#include <utility>

namespace Discretization {
    

    GhostPseudoMesh::GhostPseudoMesh(shared_ptr<list<Node*>> ghostNodesList,
                                     const shared_ptr<map<Direction, unsigned>>& ghostNodesPerDirection,
                                     const shared_ptr<map<NumericalVector<double>, Node*>>& parametricCoordToNodeMap) :
            ghostNodesList(std::move(ghostNodesList)),
            ghostNodesPerDirection(ghostNodesPerDirection),
            parametricCoordToNodeMap(parametricCoordToNodeMap) {}
    
    GhostPseudoMesh::~GhostPseudoMesh() {
        for (auto& node : *ghostNodesList){
            delete node;
            node = nullptr;
        }
    }

    
} // Discretization