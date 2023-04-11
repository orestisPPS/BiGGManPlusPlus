//
// Created by hal9000 on 4/10/23.
//

#include "GhostPseudoMesh.h"

#include <utility>

namespace Discretization {
    
    GhostPseudoMesh::GhostPseudoMesh(Mesh* targetMesh, map<Direction, unsigned>* ghostNodesPerDirection) {
        ghostedNodesMatrix = nullptr;
        parametricCoordToNodeMap = nullptr;
        this->targetMesh = nullptr;
        ghostNodesList = nullptr;
    }
    
    GhostPseudoMesh::~GhostPseudoMesh() {
        delete ghostedNodesMatrix;
        ghostedNodesMatrix = nullptr;
        delete parametricCoordToNodeMap;
        parametricCoordToNodeMap = nullptr;
    }

    void GhostPseudoMesh::initialize() {
        parametricCoordToNodeMap = createParametricCoordToNodeMap();
        ghostedNodesMatrix = createGhostedNodesMatrix();
    }

    map<vector<double>, Node*> * GhostPseudoMesh::createParametricCoordToNodeMap() {
        parametricCoordToNodeMap = new map<vector<double>, Node*>();
        for (auto node : *targetMesh->totalNodesVector)
            parametricCoordToNodeMap->insert(pair<vector<double>, Node*>
                    (node->coordinates.positionVector(Parametric), node));
        return parametricCoordToNodeMap;
    }
    
    Array<Node*>* GhostPseudoMesh::createGhostedNodesMatrix() {
        return nullptr;
    }
    
    unsigned GhostPseudoMesh::dimensions() const {
        return targetMesh->dimensions();
    }
    
    SpaceEntityType GhostPseudoMesh::space() const {
        return targetMesh->space();
    }
    
    Node* GhostPseudoMesh::node(unsigned i) {
        return nullptr;
    }
    
    Node* GhostPseudoMesh::node(unsigned i, unsigned j) {
        return nullptr;
    }
    
    Node* GhostPseudoMesh::node(unsigned i, unsigned j, unsigned k) {
        return nullptr;
    }
    
} // Discretization