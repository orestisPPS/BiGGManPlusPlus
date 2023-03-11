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
    
    unsigned Mesh::dimensions(){ }
    
    SpaceEntityType Mesh::space() { }
    
    Node* Mesh::node(unsigned i) { }
    
    Node* Mesh::node(unsigned i, unsigned j) { }
    
    Node* Mesh::node(unsigned i, unsigned j, unsigned k) { }

    void Mesh::printMesh() { }
    
    map<Position, vector<Node*>*> *Mesh::addDBoundaryNodesToMap() { }
    
    vector<Node*>* Mesh::addInternalNodesToVector() { }    
} // Discretization