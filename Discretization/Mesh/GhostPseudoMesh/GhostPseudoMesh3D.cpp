//
// Created by hal9000 on 4/11/23.
//

#include "GhostPseudoMesh3D.h"

namespace Discretization {

    GhostPseudoMesh3D::GhostPseudoMesh3D(Mesh* targetMesh, map<Direction, unsigned>* ghostNodesPerDirection) :
            GhostPseudoMesh(targetMesh, ghostNodesPerDirection) {
        this->targetMesh = targetMesh;
        this->ghostNodesPerDirection = ghostNodesPerDirection;
        ghostNodesList = new list<Node*>();
        allNodesList = new list<Node*>();
        parametricCoordToNodeMap = createParametricCoordToNodeMap();
    }

    GhostPseudoMesh3D::~GhostPseudoMesh3D() {
        delete ghostNodesList;
        delete parametricCoordToNodeMap;
        delete ghostedNodesMatrix;
        delete ghostNodesPerDirection;
    }

    Array<Node*>* GhostPseudoMesh3D::createGhostedNodesMatrix() {
        auto nodeArrayPositionI = 0;
        auto nodeArrayPositionJ = 0;
        auto nodeArrayPositionK = 0;

        auto nn1 = targetMesh->numberOfNodesPerDirection[One];
        auto nn1Ghost = ghostNodesPerDirection->at(One);
        auto nn2 = targetMesh->numberOfNodesPerDirection[Two];
        auto nn2Ghost = ghostNodesPerDirection->at(Two);
        auto nn3 = targetMesh->numberOfNodesPerDirection[Three];
        auto nn3Ghost = ghostNodesPerDirection->at(Three);
        
        auto ghostedNodesMatrix = new Array<Node*>(nn1 + 2 * nn1Ghost, nn2 + 2 * nn2Ghost, nn3 + 2 * nn3Ghost);
        for (int k = -static_cast<int>(nn3Ghost); k < static_cast<int>(nn3) + static_cast<int>(nn3Ghost); k++) {
            for (int j = -static_cast<int>(nn2Ghost); j < static_cast<int>(nn2) + static_cast<int>(nn2Ghost); j++) {
                for (int i = -static_cast<int>(nn1Ghost); i < static_cast<int>(nn1) + static_cast<int>(nn1Ghost); i++) {
                    auto parametricCoords = vector<double>{static_cast<double>(i), static_cast<double>(j), static_cast<double>(k)};
                    if (parametricCoordToNodeMap->find(parametricCoords) != parametricCoordToNodeMap->end()) {
                        auto node = parametricCoordToNodeMap->at(parametricCoords);
                        (*ghostedNodesMatrix)(nodeArrayPositionI, nodeArrayPositionJ, nodeArrayPositionK) = node;
                        allNodesList->push_back(node);
                    } else {
                        auto node = new Node();
                        node->coordinates.setPositionVector(parametricCoords, Parametric);
                        vector<double> templateCoord = {static_cast<double>(i) * targetMesh->specs->templateStepOne,
                                                        static_cast<double>(j) * targetMesh->specs->templateStepTwo,
                                                        static_cast<double>(k) * targetMesh->specs->templateStepThree};
                        // Rotate 
                        Transformations::rotate(templateCoord, targetMesh->specs->templateRotAngleOne);
                        // Shear
                        Transformations::shear(templateCoord, targetMesh->specs->templateShearOne,targetMesh->specs->templateShearTwo);
                        
                        node->coordinates.setPositionVector(templateCoord, Template);
                        (*ghostedNodesMatrix)(nodeArrayPositionI, nodeArrayPositionJ, nodeArrayPositionK) = node;
                        ghostNodesList->push_back(node);
                        allNodesList->push_back(node);
                    }
                    nodeArrayPositionI++;
                }
                nodeArrayPositionI = 0;
                nodeArrayPositionJ++;
            }
            nodeArrayPositionJ = 0;
            nodeArrayPositionK++;
        }
        return ghostedNodesMatrix;
    }

    Node* GhostPseudoMesh3D::node(unsigned i) {
        throw runtime_error("Node Not Found!");
    }

    Node* GhostPseudoMesh3D::node(unsigned i, unsigned j) {
        if (ghostedNodesMatrix != nullptr)
            return (*ghostedNodesMatrix)(i, j);
        else
            throw runtime_error("Node Not Found!");
    }

    Node* GhostPseudoMesh3D::node(unsigned i, unsigned j, unsigned k) {
        throw runtime_error("Node Not Found!");
    }
} // D
