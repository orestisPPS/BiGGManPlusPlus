//
// Created by hal9000 on 4/10/23.
//

#include "GhostPseudoMesh.h"

#include <utility>

namespace Discretization {
    

    GhostPseudoMesh::GhostPseudoMesh(list<Node*>* ghostNodesList,
                                     map<Direction, unsigned>* ghostNodesPerDirection,
                                     map<vector<double>, Node*>* parametricCoordToNodeMap) :
            ghostNodesList(ghostNodesList),
            ghostNodesPerDirection(ghostNodesPerDirection),
            parametricCoordToNodeMap(parametricCoordToNodeMap) {}
    
    GhostPseudoMesh::~GhostPseudoMesh() {
        for (auto& node : *ghostNodesList)
            delete node;
        delete ghostNodesList;
        delete ghostNodesPerDirection;
        delete parametricCoordToNodeMap;
    }

    
} // Discretization