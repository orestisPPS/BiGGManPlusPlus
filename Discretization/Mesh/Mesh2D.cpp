//
// Created by hal9000 on 3/11/23.
//

#include "Mesh2D.h"

namespace Discretization {
    
        Mesh2D::Mesh2D(Array<Node *> *nodes) : Mesh() {
            this->_nodesMatrix = nodes;
            numberOfNodesPerDirection = map<Direction, unsigned>();
            numberOfNodesPerDirection[One] = _nodesMatrix->numberOfColumns();
            numberOfNodesPerDirection[Two] = _nodesMatrix->numberOfRows();
            numberOfNodesPerDirection[Three] = _nodesMatrix->numberOfAisles();
        }
    
        Mesh2D::~Mesh2D() { }
    
        unsigned Mesh2D::dimensions() {
            return 2;
        }
    
        SpaceEntityType Mesh2D::space() {
            return Plane;
        }
    
        Node *Mesh2D::node(unsigned i) {
            if (isInitialized)
                return (*_nodesMatrix)(i);
            else
                throw runtime_error("Node Not Found!");
        }
    
        Node *Mesh2D::node(unsigned i, unsigned j) {
            if (isInitialized)
                return (*_nodesMatrix)(i, j);
            else
                throw runtime_error("Node Not Found!");
        }
    
        Node *Mesh2D::node(unsigned i, unsigned j, unsigned k) {
            if (k != 0 && isInitialized)
                return (*_nodesMatrix)(i, j);
            else
                throw runtime_error("A 2D Mesh can be considered a 3D mesh with 1 Node at Direction 3."
                                    " Third entry must be 0.");
        }
    
        void Mesh2D::printMesh() {
            cout << "Mesh2D" << endl;
            for (int i = 0 ; i < numberOfNodesPerDirection[One] ; i++) {
                for (int j = 0 ; j < numberOfNodesPerDirection[Two] ; j++) {
                    throw runtime_error("Not implemented");
                }
            }
        }
        
        map<Position, vector<Node*>*>* Mesh2D::addDBoundaryNodesToMap() {
            auto boundaryNodes = new map<Position, vector<Node*>*>();
                      
            auto *leftBoundaryNodes = new vector<Node*>(numberOfNodesPerDirection[Two]);
            auto *rightBoundaryNodes = new vector<Node*>(numberOfNodesPerDirection[Two]);
            for (int i = 0 ; i < numberOfNodesPerDirection[Two] ; i++) {
                (*leftBoundaryNodes)[i] = (*_nodesMatrix)(0, numberOfNodesPerDirection[Two] - 1 - i);
                (*rightBoundaryNodes)[i] = (*_nodesMatrix)(numberOfNodesPerDirection[Two] - 1, i);
            }
            boundaryNodes->insert( pair<Position, vector<Node*>*>(Position::Left, leftBoundaryNodes));
            boundaryNodes->insert( pair<Position, vector<Node*>*>(Position::Right, rightBoundaryNodes));
            
            auto *bottomBoundaryNodes = new vector<Node*>(numberOfNodesPerDirection[One]);
            auto *topBoundaryNodes = new vector<Node*>(numberOfNodesPerDirection[One]);
            for (int i = 0 ; i < numberOfNodesPerDirection[One] ; i++) {
                (*bottomBoundaryNodes)[i] = (*_nodesMatrix)(i, 0);
                (*topBoundaryNodes)[i] = (*_nodesMatrix)(numberOfNodesPerDirection[One] - 1 - i, numberOfNodesPerDirection[Two] - 1);
            }
            
            return boundaryNodes;
        }
        
        vector<Node*>* Mesh2D::addInternalNodesToVector() {
            auto boundaryNodes = new vector<Node*>();
            
            for (int j = 1; j < numberOfNodesPerDirection[Two] - 1; j++){
                for (int i = 1; i < numberOfNodesPerDirection[One] ; ++i) {
                    boundaryNodes->push_back((*_nodesMatrix)(i, j));
                }
            }
        }
} // 

