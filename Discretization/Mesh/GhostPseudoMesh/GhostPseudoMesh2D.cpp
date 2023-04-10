//
// Created by hal9000 on 4/11/23.
//

#include "GhostPseudoMesh2D.h"
#include "../../../LinearAlgebra/Transformations.h"

namespace Discretization {


    GhostPseudoMesh2D::GhostPseudoMesh2D(Discretization::Mesh *targetMesh,
                                         map<PositioningInSpace::Direction, unsigned int>* ghostNodesPerDirection) :
            GhostPseudoMesh(targetMesh, ghostNodesPerDirection) {
        this->targetMesh = targetMesh;
        this->ghostNodesPerDirection = ghostNodesPerDirection;
        ghostNodes = new list<Node*>();
        parametricCoordToNodeMap = createParametricCoordToNodeMap();
    }

    GhostPseudoMesh2D::~GhostPseudoMesh2D() {
        delete ghostNodes;
        delete parametricCoordToNodeMap;
        delete ghostedNodesMatrix;
        delete ghostNodesPerDirection;
    }

    Array<Node*>* GhostPseudoMesh2D::createGhostedNodesMatrix() {
        auto nodeArrayPositionI = 0;
        auto nodeArrayPositionJ = 0;
        for (unsigned j = -ghostNodesPerDirection->at(Two); j < targetMesh->numberOfNodesPerDirection[Two] + ghostNodesPerDirection->at(Two); j++) {
            for (unsigned i = -ghostNodesPerDirection->at(One); i < targetMesh->numberOfNodesPerDirection[One] + ghostNodesPerDirection->at(One); i++) {
                auto parametricCoords = vector<double>{static_cast<double>(i), static_cast<double>(j)};
                if (parametricCoordToNodeMap->find(parametricCoords) != parametricCoordToNodeMap->end()) {
                    auto node = parametricCoordToNodeMap->at(parametricCoords);
                    (*ghostedNodesMatrix)(nodeArrayPositionI, nodeArrayPositionJ) = node;
                } else {
                    auto node = new Node();
                    node->coordinates.setPositionVector(parametricCoords, Parametric);
                    vector<double> templateCoord = {static_cast<double>(i) * targetMesh->specs->templateStepOne,
                                                    static_cast<double>(j) * targetMesh->specs->templateStepTwo};
                    // Rotate 
                    Transformations::rotate(templateCoord, targetMesh->specs->templateRotAngleOne);
                    // Shear
                    Transformations::shear(templateCoord, targetMesh->specs->templateShearOne,targetMesh->specs->templateShearTwo);
                    
                    node->coordinates.setPositionVector(templateCoord, Template);
                    (*ghostedNodesMatrix)(nodeArrayPositionI, nodeArrayPositionJ) = node;
                    ghostNodes->push_back(node);
                }
                nodeArrayPositionI++;
            }
            nodeArrayPositionJ++;
            nodeArrayPositionI = 0;
        }
        return ghostedNodesMatrix;
    }

    Node* GhostPseudoMesh2D::node(unsigned i) {
        throw runtime_error("Node Not Found!");
    }

    Node* GhostPseudoMesh2D::node(unsigned i, unsigned j) {
        if (ghostedNodesMatrix != nullptr)
            return (*ghostedNodesMatrix)(i, j);
        else
            throw runtime_error("Node Not Found!");
    }

    Node* GhostPseudoMesh2D::node(unsigned i, unsigned j, unsigned k) {
        throw runtime_error("Node Not Found!");
    }
} // Discretization// Discretization