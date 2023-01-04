//
// Created by hal9000 on 12/17/22.
//
#pragma once

#include "../Node/Node.h"
#include<vector>
#include "../../Primitives/Array.h"
using namespace Discretization;
using namespace LinearAlgebra;

namespace Discretization {

    class Mesh {
    public:
        //Mesh(Array<Node *> *nodes, map<Direction, int> numberOfNodesPerDirection);
        Mesh(Array<Node *> *nodes);
        
        ~Mesh();
        
        map<Direction, int> numberOfNodesPerDirection;

        map<Position, list<Node *>*> *boundaryNodes;
        
        map<int, Node *> *nodeMap;

        int TotalNodes();

        int MeshDimensions();
        
        Node *node(int i);

        Node *node(int i, int j);

        Node *node(int i, int j, int k);
    
    private:
        Array<Node *> *_nodesMatrix;
        
        map<Position, list<Node*>*> *CreateBoundaries();

        map<Position, list<Node*>*> *Create1DBoundaries();

        map<Position, list<Node*>*> *Create2DBoundaries();

        map<Position, list<Node*>*> *Create3DBoundaries();
    };
}