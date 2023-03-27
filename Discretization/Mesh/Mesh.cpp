//
// Created by hal9000 on 12/17/22.
//

#include "Mesh.h"

#include <utility>
using namespace  Discretization;

namespace Discretization {
    
    Mesh::Mesh() { }
        
    Mesh::~Mesh() {
        delete _nodesMatrix;
        _nodesMatrix = nullptr;
        delete boundaryNodes;
        boundaryNodes = nullptr;
    }
        
    unsigned Mesh::totalNodes() {
        if (isInitialized)
            return _nodesMatrix->size();
        else
            throw std::runtime_error("Mesh has not been initialized");
    }
    
    Node* Mesh::nodeFromID(unsigned ID) {
        if (isInitialized)
            return _nodesMatrix->at(ID);
        else
            throw std::runtime_error("Mesh has not been initialized");
    }
    

    
    unsigned Mesh::dimensions(){ return 0;}
    
    SpaceEntityType Mesh::space() {
        return NullSpace;
    }
    Node* Mesh::node(unsigned i) {
        return nullptr;
     }
    
    Node* Mesh::node(unsigned i, unsigned j) {
        return nullptr;
    }
    
    Node* Mesh::node(unsigned i, unsigned j, unsigned k) {
        return nullptr;
    }

    void Mesh::printMesh() { }
    
    void Mesh::initialize() {
        isInitialized = true;
        createNumberOfNodesPerDirectionMap();
        categorizeNodes();
    }
    
    map<Position, vector<Node*>*>* Mesh::addDBoundaryNodesToMap() {
        return nullptr;
    }
    
    vector<Node*>* Mesh::addInternalNodesToVector() {
        return nullptr;
    }    
    
    
    void createNumberOfNodesPerDirectionMap() { }

    void Mesh::categorizeNodes() {
        if (isInitialized) {
            boundaryNodes = addDBoundaryNodesToMap();
            internalNodes = addInternalNodesToVector();
        }
        else
            throw std::runtime_error("Mesh has not been initialized");
    }
    
    void Mesh::createNumberOfNodesPerDirectionMap() {
        if (isInitialized) {
            numberOfNodesPerDirection = map<Direction, unsigned>();
            numberOfNodesPerDirection[One] = _nodesMatrix->numberOfColumns();
            numberOfNodesPerDirection[Two] = _nodesMatrix->numberOfRows();
            numberOfNodesPerDirection[Three] = _nodesMatrix->numberOfAisles();
        }
        else
            throw std::runtime_error("Mesh has not been initialized");
    }
    
    void Mesh::cleanMeshDataStructures() {
        //search all boundaryNodes map and deallocate all vector pointer values
        for (auto &boundary : *boundaryNodes) {
            delete boundary.second;
            boundary.second = nullptr;
        }
        delete boundaryNodes;
        boundaryNodes = nullptr;

        //Deallocate internalNodes vector
        delete internalNodes;
        internalNodes = nullptr;
    }
} // Discretization