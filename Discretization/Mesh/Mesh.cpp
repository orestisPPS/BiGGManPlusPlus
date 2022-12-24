//
// Created by hal9000 on 12/17/22.
//

#include "Mesh.h"
using namespace  Discretization;

namespace Discretization {
    
    Mesh::Mesh(Array<Node *> *nodes, map<Direction, int> numberOfNodesPerDirection) {
        this->_nodesMatrix = nodes;
        numberOfNodesPerDirection = numberOfNodesPerDirection;
        boundaryNodes = CreateBoundaries();
        //nodeMap = new map<>
    }
        
    Mesh::~Mesh() {
        delete _nodesMatrix;
        _nodesMatrix = nullptr;
        delete nodeMap;
        nodeMap = nullptr;
        delete boundaryNodes;
        boundaryNodes = nullptr;
    }
    
    int Mesh::TotalNodes() {
        if (_nodesMatrix != nullptr)
            return nodeMap->size();
        else
            return 0;
    }
    
    int Mesh::MeshDimensions() {
        return numberOfNodesPerDirection.size();
    }
    
    Node* Mesh::node(int i) {
        if (_nodesMatrix != nullptr)
            return _nodesMatrix->element(i);
        else
            throw runtime_error("Node Not Found. You search for a 1D node in a" + to_string(MeshDimensions()) + "D mesh.");
    }
        
    
    Node* Mesh::node(int i, int j) {
        if (_nodesMatrix != nullptr)
            return _nodesMatrix->element(i, j);
        else
            throw runtime_error("Node Not Found. You search for a 2D node in a" + to_string(MeshDimensions()) + "D mesh.");
        }
    
    Node* Mesh::node(int i, int j, int k) {
        if (_nodesMatrix != nullptr)
            return _nodesMatrix->element(i, j, k);
        else
            throw runtime_error("Node Not Found. You search for a 3D node in a" + to_string(MeshDimensions()) + "D mesh.");
    }
    
    map<Position, list<Node*>*>* Mesh::CreateBoundaries() {
        switch (MeshDimensions()) {
            case 1:
                return Create1DBoundaries();
            case 2:
                return Create2DBoundaries();
            default:
                return Create3DBoundaries();
        }
    }

    map<Position, list<Node*>*> *Mesh::Create1DBoundaries() {
        auto boundaries = new map<Position, list<Node*>*>();
        auto leftBoundary = new list<Node*>();
        auto rightBoundary = new list<Node*>();
        rightBoundary->push_back(Mesh::node(0));
        leftBoundary->push_back(Mesh::node(Mesh::numberOfNodesPerDirection[Direction::One] - 1));
        boundaries->insert( pair<Position, list<Node*>*>(Position::Left, leftBoundary));
        return boundaries;   
    }
    
    map<Position, list<Node*>*> *Mesh::Create2DBoundaries() {
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

    map<Position, list<Node*>*> *Mesh::Create3DBoundaries() {
        throw runtime_error ("Not implemented");
    }
    
    
    
} // Discretization