//
// Created by hal9000 on 12/17/22.
//

#include "Mesh.h"

#include <utility>
using namespace  Discretization;

namespace Discretization {
    
    Mesh::Mesh() {
        _nodesMatrix = nodes;

        
        boundaryNodes = listBoundaryNodes1D();
        //nodeMap = new map<>
    }
        
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
    
    
    map<Position, list<Node*>*> *Mesh::listBoundaryNodes3D() {
        auto boundaries = new map<Position, list<Node*>*>();
        
        auto leftBoundary = new list<Node*>();
        auto rightBoundary = new list<Node*>();
        for (int i = 0; i < Mesh::numberOfNodesPerDirection[Direction::Two]; i++) {
            leftBoundary->push_back(Mesh::node(0, numberOfNodesPerDirection[Direction::One] - 1 - i));
            rightBoundary->push_back(Mesh::node(Mesh::numberOfNodesPerDirection[Direction::One] - 1, i));
        }
        boundaries->insert( pair<Position, list<Node*>*>(Position::Left, leftBoundary));
        boundaries->insert( pair<Position, list<Node*>*>(Position::Right, rightBoundary));
        
        auto topBoundary = new list<Node*>();
        auto bottomBoundary = new list<Node*>();
        for (int i = 0; i < Mesh::numberOfNodesPerDirection[Direction::One]; i++) {
            bottomBoundary->push_back(Mesh::node(i, 0));
            topBoundary->push_back(Mesh::node(numberOfNodesPerDirection[Direction::One] - 1 - i, Mesh::numberOfNodesPerDirection[Direction::Two] - 1));
        }
        boundaries->insert( pair<Position, list<Node*>*>(Position::Top, topBoundary));
        boundaries->insert( pair<Position, list<Node*>*>(Position::Bottom, bottomBoundary));
        
        return boundaries;
    }

    
    
    
    
} // Discretization