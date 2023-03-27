//
// Created by hal9000 on 3/11/23.
//

#include "Mesh1D.h"

namespace Discretization {
    
    Mesh1D::Mesh1D(Array<Node *> *nodes) : Mesh(){
        this->_nodesMatrix = nodes;
        initialize();
    }
    
    Mesh1D::~Mesh1D() {
        //Deallocate all node pointers of the mesh
        for (int i = 0; i < numberOfNodesPerDirection[One] ; ++i) {
            delete (*_nodesMatrix)(i);
            (*_nodesMatrix)(i) = nullptr;
        }
        delete _nodesMatrix;
        _nodesMatrix = nullptr;
        
        cleanMeshDataStructures();
        
    }
    
    unsigned Mesh1D::dimensions() {
        return 1;
    }
    
    SpaceEntityType Mesh1D::space() {
        return Axis;
    }

    Node* Mesh1D::node(unsigned i) {
        if (_nodesMatrix != nullptr)
            return (*_nodesMatrix)(i);
        else
            throw runtime_error("Node Not Found!");
    }

    Node* Mesh1D::node(unsigned i, unsigned j) {
        if (j != 0 && isInitialized)
            return (*_nodesMatrix)(i);
        else 
            throw runtime_error("A 1D Mesh can be considered a 2D mesh with 1 Node at Direction 2."
                                " Second entry must be 0.");
    }
    
    Node* Mesh1D::node(unsigned i, unsigned j, unsigned k) {
        if (j != 1 && k != 0 && isInitialized)
            return (*_nodesMatrix)(i);
        else
            throw runtime_error("A 1D Mesh can be considered a 3D mesh with 1 Node at Directions 2 and 3."
                                " Second and third entries must be 0.");
    }
    
    void Mesh1D::printMesh() {
        cout << "Mesh1D" << endl;
        for (int i = 0 ; i < numberOfNodesPerDirection[Direction::One] ; i++) {
            (*_nodesMatrix)(i)->printNode();
        }
    }
    

    
    map<Position, vector<Node*>*> *Mesh1D::addDBoundaryNodesToMap() {
        auto boundaries = new map<Position, vector<Node*>*>();
        auto leftBoundary = new vector<Node*>(1);
        auto rightBoundary = new vector<Node*>(1);
        leftBoundary->push_back(Mesh::node(0));
        rightBoundary->push_back(Mesh::node(Mesh::numberOfNodesPerDirection[Direction::One] - 1));
        boundaries->insert( pair<Position, vector<Node*>*>(Position::Left, leftBoundary));
        return boundaries;
    }
    
    vector<Node*>* Mesh1D::addInternalNodesToVector() {
        auto internalNodes = new vector<Node*>();
        for (int i = 1; i < numberOfNodesPerDirection[Direction::One] - 1; i++) {
            internalNodes->push_back(Mesh::node(i));
        }
        return internalNodes;
    }
    
    
} // Discretization