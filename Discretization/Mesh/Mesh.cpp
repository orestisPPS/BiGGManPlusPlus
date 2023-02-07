//
// Created by hal9000 on 12/17/22.
//

#include "Mesh.h"

#include <utility>
using namespace  Discretization;

namespace Discretization {
    
    Mesh::Mesh(Array<Node *> *nodes) {
        _nodesMatrix = nodes;
        numberOfNodesPerDirection = map<Direction, unsigned>();
        numberOfNodesPerDirection[Direction::One] = _nodesMatrix->numberOfColumns();
        numberOfNodesPerDirection[Direction::Two] = _nodesMatrix->numberOfRows();
        numberOfNodesPerDirection[Direction::Three] = _nodesMatrix->numberOfAisles();
        
        boundaryNodes = CreateBoundaries();
        //nodeMap = new map<>
    }
        
    Mesh::~Mesh() {
        delete _nodesMatrix;
        _nodesMatrix = nullptr;
        delete boundaryNodes;
        boundaryNodes = nullptr;
    }
    const SpaceEntityType& Mesh::space() {
        if (numberOfNodesPerDirection[Direction::One] > 1 && numberOfNodesPerDirection[Direction::Two] > 1 && numberOfNodesPerDirection[Direction::Three] > 1)
            return Volume;
        else if (numberOfNodesPerDirection[Direction::One] > 1 && numberOfNodesPerDirection[Direction::Two] > 1)
            return Plane;
        else if (numberOfNodesPerDirection[Direction::One] > 1)
            return Axis;
        else
            throw runtime_error("Mesh has no nodes");
    }
    const unsigned& Mesh::totalNodes() const {
        if (_nodesMatrix != nullptr)
            return _totalNodes;
        else
            throw std::runtime_error("Mesh has not been initialized");
    }
    
    const unsigned& Mesh::dimensions() const  {
        if (_nodesMatrix != nullptr)
            return _dimensions;
        else
            throw std::runtime_error("Mesh has not been initialized");
    }
    
    Node* Mesh::node(unsigned i) {
        if (_nodesMatrix != nullptr)
            return (*_nodesMatrix)(i);
        else
            throw runtime_error("Node Not Found. You search for a 1D node in a" + to_string(dimensions()) + "D mesh.");
    }
            
    Node* Mesh::node(unsigned i, unsigned j) {
        if (_nodesMatrix != nullptr)
            return (*_nodesMatrix)(i, j);
        else
            throw runtime_error("Node Not Found. You search for a 2D node in a" + to_string(dimensions()) + "D mesh.");
        }
    
    Node* Mesh::node(unsigned i, unsigned j, unsigned k) {
        if (_nodesMatrix != nullptr)
            return (*_nodesMatrix)(i, j, k);
        else
            throw runtime_error("Node Not Found. You search for a 3D node in a" + to_string(dimensions()) + "D mesh.");
    }
    
    void Mesh::getSpatialProperties(map<Direction, unsigned> numberOfNodesPerDirection, unsigned dimensions, unsigned totalNodes) {
        if (!_isInitialized) {
            ///CHHANGES HERE
            /// WAKE UP GREEK MAN!!!
            this->numberOfNodesPerDirection = std::move(numberOfNodesPerDirection);
            _dimensions = dimensions;
            _totalNodes = totalNodes;
            _isInitialized = true;
        }
        else
            throw runtime_error("What the fuck are you doing? Initializing a mesh twice?");
    }
    
/*    void Mesh::printMesh() {
        if (dimensions() == 1){
            cout<<"Dimensions : 1"<<endl;
            cout<<"Number of Nodes : " << totalNodes()<<endl;
            for (int i = 0; i < totalNodes(); ++i) {
                cout<<"i = "<< i << " : Global id : "<<  node(i)->id.global << " : Boundary id : "<<  node(i)->id.boundary << " : Internal id : "<<  node(i)->id.internal;
            }
        }
        else if(dimensions() == 3)


        
        
        
        
    }*/
    
    
    
    
    map<Position, list<Node*>*>* Mesh::CreateBoundaries() {
        switch (dimensions()) {
            case 1:
                return Create1DBoundaries();
            case 2:
                return Create2DBoundaries();
                
            //default:
                //return Create3DBoundaries();
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
    
    Node* Mesh::nodeFromID(unsigned int ID) {
/*        if (dimensions() == 1)
            return node(ID);
        else if (dimensions() == 2)
            
        else*/
    }
    
    
    
    
    
} // Discretization