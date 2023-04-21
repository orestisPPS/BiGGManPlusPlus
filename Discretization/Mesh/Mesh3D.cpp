//
// Created by hal9000 on 3/11/23.
//

#include "Mesh3D.h"

namespace Discretization {
    
    Mesh3D::Mesh3D(Array<Node*>* nodes) : Mesh(){
        this->_nodesMatrix = nodes;
        initialize();
        _nodesMap = createNodesMap();
    }
    
    Mesh3D::~Mesh3D() {
        for (int i = 0; i < numberOfNodesPerDirection[One] ; ++i)
            for (int j = 0; j < numberOfNodesPerDirection[Two] ; ++j)
                for (int k = 0; k < numberOfNodesPerDirection[Three] ; ++k){
                delete (*_nodesMatrix)(i, j, k);
                (*_nodesMatrix)(i, j, k) = nullptr;
            }
        delete _nodesMatrix;
        _nodesMatrix = nullptr;
        
        cleanMeshDataStructures();
    }
    
    unsigned Mesh3D::dimensions() {
        return 3;
    }
    
    SpaceEntityType Mesh3D::space() {
        return Volume;
    }

    vector<Direction> Mesh3D::directions() {
        return {One, Two, Three};
    }
    
    Node* Mesh3D::node(unsigned i) {
        if (isInitialized)
            return (*_nodesMatrix)(i);
        else
            throw runtime_error("Node Not Found!");
    }
    
    Node* Mesh3D::node(unsigned i, unsigned j) {
        if (isInitialized)
            return (*_nodesMatrix)(i, j);
        else
            throw runtime_error("Node Not Found!");
    }
    
    Node* Mesh3D::node(unsigned i, unsigned j, unsigned k) {
        if (isInitialized)
            return (*_nodesMatrix)(i, j, k);
        else
            throw runtime_error("Node Not Found!");
    }
    
    void Mesh3D::printMesh() {
        for (int k = 0 ; k < numberOfNodesPerDirection[Three] ; k++)
            for (int j = 0 ; j < numberOfNodesPerDirection[Two] ; j++)
                for (int i = 0 ; i < numberOfNodesPerDirection[One] ; i++) {
                    (*_nodesMatrix)(i, j, k)->printNode();
                }   
    }
    
    map<Position, vector<Node*>*>* Mesh3D::addDBoundaryNodesToMap() {
        auto boundaryNodes = new map<Position, vector<Node*>*>();
        
        auto bottomNodes = new vector<Node*>();
        auto topNodes = new vector<Node*>();
        for (int i = 0 ; i < numberOfNodesPerDirection[Two] ; i++) {
            for (int j = 0 ; j < numberOfNodesPerDirection[One] ; j++) {
                bottomNodes->push_back((*_nodesMatrix)(i, j, 0));
                topNodes->push_back((*_nodesMatrix)(i, j, numberOfNodesPerDirection[Three] - 1));
            }
        }
        boundaryNodes->insert(pair<Position, vector<Node*>*>(Bottom, bottomNodes));
        boundaryNodes->insert(pair<Position, vector<Node*>*>(Top, topNodes));
        
        auto leftNodes = new vector<Node*>();
        auto rightNodes = new vector<Node*>();
        for (int i = 0 ; i < numberOfNodesPerDirection[Three] ; i++) {
            for (int j = 0 ; j < numberOfNodesPerDirection[Two] ; j++) {
                leftNodes->push_back((*_nodesMatrix)(0, j, i));
                rightNodes->push_back((*_nodesMatrix)(numberOfNodesPerDirection[One] - 1, j, i));
            }
        }
        boundaryNodes->insert(pair<Position, vector<Node*>*>(Left, leftNodes));
        boundaryNodes->insert(pair<Position, vector<Node*>*>(Right, rightNodes));
    
        auto frontNodes = new vector<Node*>();
        auto backNodes = new vector<Node*>();
        for (int i = 0 ; i < numberOfNodesPerDirection[Three] ; i++) {
            for (int j = 0 ; j < numberOfNodesPerDirection[One] ; j++) {
                frontNodes->push_back((*_nodesMatrix)(j, 0, i));
                backNodes->push_back((*_nodesMatrix)(j, numberOfNodesPerDirection[Two] - 1, i));
            }
        }
        boundaryNodes->insert(pair<Position, vector<Node*>*>(Front, frontNodes));
        boundaryNodes->insert(pair<Position, vector<Node*>*>(Back, backNodes));
        
        return boundaryNodes;
    }
    
    vector<Node*>* Mesh3D::addInternalNodesToVector() {
        auto internalNodes = new vector<Node*>();
        for (int k = 1; k < numberOfNodesPerDirection[Three] - 1; k++){
            for (int j = 1; j < numberOfNodesPerDirection[Two] - 1; j++){
                for (int i = 1; i < numberOfNodesPerDirection[One] - 1; ++i) {
                    internalNodes->push_back((*_nodesMatrix)(i, j, k));
                }
            }
        }
        return internalNodes;
    }

    vector<Node*>* Mesh3D::addTotalNodesToVector() {
        auto totalNodes = new vector<Node*>(_nodesMatrix->size());
        for (int k = 0; k < numberOfNodesPerDirection[Three]; k++){
            for (int j = 0; j < numberOfNodesPerDirection[Two]; j++){
                for (int i = 0; i < numberOfNodesPerDirection[One]; ++i) {
                    totalNodes->push_back((*_nodesMatrix)(i, j, k));
                }
            }
        }
        return totalNodes;      
    }
    
    GhostPseudoMesh* Mesh3D::createGhostPseudoMesh(unsigned ghostLayerDepth) {
        auto ghostNodesPerDirection = createNumberOfGhostNodesPerDirectionMap(ghostLayerDepth);

        auto ghostNodesList = new list<Node*>();

        // Parametric coordinate 1 of nodes in the new ghost mesh
        auto nodeArrayPositionI = 0;
        // Parametric coordinate 2 of nodes in the new ghost mesh
        auto nodeArrayPositionJ = 0;
        // Parametric coordinate 3 of nodes in the new ghost mesh
        auto nodeArrayPositionK = 0;
        
        auto nn1 = numberOfNodesPerDirection[One];
        auto nn1Ghost = ghostNodesPerDirection->at(One);
        auto nn2 = numberOfNodesPerDirection[Two];
        auto nn2Ghost = ghostNodesPerDirection->at(Two);
        auto nn3 = numberOfNodesPerDirection[Three];
        auto nn3Ghost = ghostNodesPerDirection->at(Three);

        auto parametricCoordToNodeMap =  createParametricCoordToNodesMap();
        
        for (int k = -static_cast<int>(nn3Ghost); k < static_cast<int>(nn3) + static_cast<int>(nn3Ghost); k++){
            for (int j = -static_cast<int>(nn2Ghost); j < static_cast<int>(nn2) + static_cast<int>(nn2Ghost); j++) {
                for (int i = -static_cast<int>(nn1Ghost); i < static_cast<int>(nn1) + static_cast<int>(nn1Ghost); ++i) {
                    auto parametricCoords = vector<double>{static_cast<double>(i), static_cast<double>(j),
                                                           static_cast<double>(k)};
                    // If node is inside the original mesh add it to the ghost mesh Array
                    if (parametricCoordToNodeMap->find(parametricCoords) == parametricCoordToNodeMap->end()) {
                        auto node = new Node();
                        node->coordinates.setPositionVector(parametricCoords, Parametric);
                        vector<double> templateCoord = {static_cast<double>(i) * specs->templateStepOne,
                                                        static_cast<double>(j) * specs->templateStepTwo};
                        // Rotate 
                        Transformations::rotate(templateCoord, specs->templateRotAngleOne);
                        // Shear
                        Transformations::shear(templateCoord, specs->templateShearOne, specs->templateShearTwo);

                        node->coordinates.setPositionVector(templateCoord, Template);
                        ghostNodesList->push_back(node);
                    }
                    nodeArrayPositionI++;
                }
                nodeArrayPositionI = 0;
                nodeArrayPositionJ++;
            }
            nodeArrayPositionJ = 0;
            nodeArrayPositionK++;
        }
        return new GhostPseudoMesh(ghostNodesList, ghostNodesPerDirection, parametricCoordToNodeMap);
    }
    
    map<vector<double>, Node*>* Mesh3D::createParametricCoordToNodesMap() {
        auto parametricCoordToNodeMap = new map<vector<double>, Node*>();
        for (auto& node : *totalNodesVector) {
            auto parametricCoords = node->coordinates.positionVector(Parametric);
            parametricCoordToNodeMap->insert(pair<vector<double>, Node*>(parametricCoords, node));
        }
        return parametricCoordToNodeMap;
    }
    
    Metrics* Mesh3D::calculateNodeMetrics(Discretization::Node *node,
                                          PositioningInSpace::CoordinateType coordinateSystem) {
        return nullptr;
    }

    


} // Discretization