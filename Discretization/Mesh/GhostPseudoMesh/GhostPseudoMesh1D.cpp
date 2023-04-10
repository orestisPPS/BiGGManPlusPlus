//
// Created by hal9000 on 4/11/23.
//

#include "GhostPseudoMesh1D.h"

namespace Discretization {

    GhostPseudoMesh1D::GhostPseudoMesh1D(Discretization::Mesh *targetMesh,
                                         map<PositioningInSpace::Direction, unsigned int>* ghostNodesPerDirection) :
                                         GhostPseudoMesh(targetMesh, ghostNodesPerDirection) {
        this->targetMesh = targetMesh;
        this->ghostNodesPerDirection = ghostNodesPerDirection;
        ghostNodes = new list<Node*>();
        parametricCoordToNodeMap = createParametricCoordToNodeMap();
    }
    
    GhostPseudoMesh1D::~GhostPseudoMesh1D() {
        delete ghostNodes;
        delete parametricCoordToNodeMap;
        delete ghostedNodesMatrix;
        delete ghostNodesPerDirection;
    }
    
    Array<Node*>* GhostPseudoMesh1D::createGhostedNodesMatrix() {
        auto nodeArrayPositionI = 0;
        for (unsigned i = -ghostNodesPerDirection->at(One); i < targetMesh->numberOfNodesPerDirection[One] + ghostNodesPerDirection->at(One); i++) {
            auto parametricCoords = vector<double>{static_cast<double>(i) };
            if (parametricCoordToNodeMap->find(parametricCoords) != parametricCoordToNodeMap->end()) {
                auto node = parametricCoordToNodeMap->at(parametricCoords);
                (*ghostedNodesMatrix)(nodeArrayPositionI) = node;
            } else {
                auto node = new Node();
                node->coordinates.setPositionVector(parametricCoords, Parametric);
                node->coordinates.setPositionVector({static_cast<double>(i) *targetMesh->specs->templateStepOne}, Template);
                (*ghostedNodesMatrix)(nodeArrayPositionI) = node;
                ghostNodes->push_back(node);
            }

        }
        
        return ghostedNodesMatrix;
    }
    
    Node* GhostPseudoMesh1D::node(unsigned i) {
        if (ghostedNodesMatrix != nullptr)
            return (*ghostedNodesMatrix)(i);
        else
            throw runtime_error("Node Not Found!");
    }
    
    Node* GhostPseudoMesh1D::node(unsigned i, unsigned j) {
        throw runtime_error("Node Not Found!");
    }
    
    Node* GhostPseudoMesh1D::node(unsigned i, unsigned j, unsigned k) {
        throw runtime_error("Node Not Found!");
    }
} // Discretization