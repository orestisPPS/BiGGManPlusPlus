//
// Created by hal9000 on 12/17/22.
//
#pragma once

#include "../Node/Node.h"
#include<vector>
#include "../../LinearAlgebra/Array.h"
using namespace Discretization;
using namespace LinearAlgebra;

namespace Discretization {

    class Mesh {
    public:
        //Mesh(Array<Node *> *nodes, map<Direction, int> numberOfNodesPerDirection);
        Mesh(Array<Node *> *nodes);
        
        ~Mesh();

        const SpaceEntityType& space();
        
        //map<Direction, unsigned > *numberOfNodesPerDirection;
        map<Direction, unsigned > numberOfNodesPerDirection;

        map<Position, vector<Node*>*>* boundaryNodes;
        
        vector<Node*>* internalNodes;

        const unsigned &totalNodes() const;

        const unsigned &dimensions() const;
        
        Node *node(unsigned i);

        Node *node(unsigned i, unsigned j);

        Node *node(unsigned i, unsigned j, unsigned k);
        
        Node *nodeFromID(unsigned ID);
        
        //Gets called by the mesh preprocessor to initiate space, numberOfNodesPerDirection, and dimensions
        void getSpatialProperties(map<Direction, unsigned> numberOfNodesPerDirection, unsigned dimensions, unsigned totalNodes, SpaceEntityType space);
        
        void printMesh();
    
    private:
        
        SpaceEntityType _space;
        
        unsigned _dimensions;
        
        unsigned _totalNodes;
        
        bool _isInitialized;
        
        Array<Node *> *_nodesMatrix;
        
        //Adds the boundary nodes of the 1D mesh to a map pointer of enum Position and vector pointers of node pointers
        map<Position, vector<Node*>*> *add1DBoundaryNodesToMap();
        
        //Adds the boundary nodes of the 2D mesh to a map pointer of enum Position and vector pointers of node pointers
        map<Position, vector<Node*>*> *add2DBoundaryNodesToMap();
        
        //Adds the boundary nodes of the 3D mesh to a map pointer of enum Position and vector pointers of node pointers
        map<Position, vector<Node*>*> *add3DBoundaryNodesToMap();
        
        //Adds the internal nodes of the mesh to a vector pointer of node pointers
        vector<Node*>* addInternalNodesToVector();
        
    };
}