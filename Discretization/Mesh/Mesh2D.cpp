//
// Created by hal9000 on 3/11/23.
//

#include "Mesh2D.h"

namespace Discretization {
    
        Mesh2D::Mesh2D(Array<Node *> *nodes) : Mesh() {
            this->_nodesMatrix = nodes;
            initialize();      
        }
    
        Mesh2D::~Mesh2D() {
            //Deallocate all node pointers of the mesh
            for (int i = 0; i < numberOfNodesPerDirection[One] ; ++i)
                for (int j = 0; j < numberOfNodesPerDirection[Two] ; ++j){
                delete (*_nodesMatrix)(i, j);
                (*_nodesMatrix)(i, j) = nullptr;
            }
            delete _nodesMatrix;
            _nodesMatrix = nullptr;
            
            cleanMeshDataStructures();
        }
    
        unsigned Mesh2D::dimensions() {
            return 2;
        }
    
        SpaceEntityType Mesh2D::space() {
            return Plane;
        }
    
        Node *Mesh2D::node(unsigned i) {
            if (isInitialized)
                return (*_nodesMatrix)(i, 0);
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
            cout << "Number of Nodes : " << totalNodes() << endl;
            for (int i = 0; i < numberOfNodesPerDirection[Two] ; ++i) {
                for (int j = 0; j < numberOfNodesPerDirection[One] ; ++j) {
                    cout << "(i, j) : (" << j << ", " << i << ")" << endl;
                    cout << "ID (Global, Boundary, Internal) : (" << (*(*_nodesMatrix)(j, i)->id.global) << ", "
                                                                 << (*(*_nodesMatrix)(j, i)->id.boundary) << ", "
                                                                 << (*(*_nodesMatrix)(j, i)->id.internal) << ")" << endl;

                    cout <<" " << endl;
                }

            }
        }
        
        map<Position, vector<Node*>*>* Mesh2D::addDBoundaryNodesToMap() {
            auto boundaryNodes = new map<Position, vector<Node*>*>();
                      
            auto *leftBoundaryNodes = new vector<Node*>(numberOfNodesPerDirection[Two]);
            auto *rightBoundaryNodes = new vector<Node*>(numberOfNodesPerDirection[Two]);
            for (int i = 0 ; i < numberOfNodesPerDirection[Two] ; i++) {
                (*leftBoundaryNodes)[i] = (*_nodesMatrix)(0, i);
                (*rightBoundaryNodes)[i] = (*_nodesMatrix)(numberOfNodesPerDirection[One] - 1, i);
            }
            boundaryNodes->insert( pair<Position, vector<Node*>*>(Position::Left, leftBoundaryNodes));
            boundaryNodes->insert( pair<Position, vector<Node*>*>(Position::Right, rightBoundaryNodes));
            
            auto *bottomBoundaryNodes = new vector<Node*>(numberOfNodesPerDirection[One]);
            auto *topBoundaryNodes = new vector<Node*>(numberOfNodesPerDirection[One]);
            for (int i = 0 ; i < numberOfNodesPerDirection[One] ; i++) {
                (*bottomBoundaryNodes)[i] = (*_nodesMatrix)(i, 0);
                (*topBoundaryNodes)[i] = (*_nodesMatrix)(i, numberOfNodesPerDirection[Two] - 1);
            }
            boundaryNodes->insert( pair<Position, vector<Node*>*>(Position::Bottom, bottomBoundaryNodes));
            boundaryNodes->insert( pair<Position, vector<Node*>*>(Position::Top, topBoundaryNodes));
            
            return boundaryNodes;
        }
        
        vector<Node*>* Mesh2D::addInternalNodesToVector() {
            auto internalNodes = new vector<Node*>();
            for (int j = 1; j < numberOfNodesPerDirection[Two] - 1; j++){
                for (int i = 1; i < numberOfNodesPerDirection[One] - 1; ++i) {
                    internalNodes->push_back((*_nodesMatrix)(i, j));
                }
            }
            return internalNodes;
        }

        vector<Node*>* Mesh2D::addTotalNodesToVector() {
            auto totalNodes = new vector<Node*>(_nodesMatrix->size());
            for (int j = 0; j < numberOfNodesPerDirection[Two] ; j++){
                for (int i = 0; i < numberOfNodesPerDirection[One] ; ++i) {
                    (*totalNodes)[i + j * numberOfNodesPerDirection[One]] = (*_nodesMatrix)(i, j);
                }
            }
            return totalNodes;
            //print node id

            
        }
        
}
        

