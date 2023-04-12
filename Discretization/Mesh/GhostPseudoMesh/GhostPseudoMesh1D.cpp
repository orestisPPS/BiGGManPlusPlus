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
        ghostNodesList = new list<Node*>();
        allNodesList = new list<Node*>();
        
        parametricCoordToNodeMap = createParametricCoordToNodeMap();
    }
    
    GhostPseudoMesh1D::~GhostPseudoMesh1D() {
        delete ghostNodesList;
        delete parametricCoordToNodeMap;
        delete ghostedNodesMatrix;
        delete ghostNodesPerDirection;
    }
    
    Array<Node*>* GhostPseudoMesh1D::createGhostedNodesMatrix() {
        auto nodeArrayPositionI = 0;
        auto nn1 = targetMesh->numberOfNodesPerDirection[One];
        auto nn1Ghost = ghostNodesPerDirection->at(One);
        auto ghostedNodesMatrix = new Array<Node*>(nn1 + 2 * nn1Ghost);
        for (int i = -static_cast<int>(nn1Ghost); i < static_cast<int>(nn1) + static_cast<int>(nn1Ghost); i++) {
            auto parametricCoords = vector<double>{static_cast<double>(i)};
            if (parametricCoordToNodeMap->find(parametricCoords) != parametricCoordToNodeMap->end()) {
                auto node = parametricCoordToNodeMap->at(parametricCoords);
                (*ghostedNodesMatrix)(nodeArrayPositionI) = node;
                allNodesList->push_back(node);
            } else {
                auto node = new Node();
                node->coordinates.setPositionVector(parametricCoords, Parametric);
                vector<double> templateCoord = {static_cast<double>(i) * targetMesh->specs->templateStepOne};
                node->coordinates.setPositionVector(templateCoord, Template);
                (*ghostedNodesMatrix)(nodeArrayPositionI) = node;
                ghostNodesList->push_back(node);
                allNodesList->push_back(node);
            }
            nodeArrayPositionI++;
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