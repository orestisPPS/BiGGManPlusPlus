//
// Created by hal9000 on 12/17/22.
//
#pragma once

#include "../Node/Node.h"
#include<vector>
#include "../../LinearAlgebra/Array.h"
#include "../../PositioningInSpace/SpaceCharacteristics.h"

using namespace Discretization;
using namespace LinearAlgebra;

namespace Discretization {

    class Mesh {
    public:
        //Mesh(Array<Node *> *nodes, map<Direction, int> numberOfNodesPerDirection);
        Mesh(Array<Node *> *nodes);
        
        ~Mesh();
        
        SpaceCharacteristics *spaceCharacteristics;
        
        map<Direction, unsigned > numberOfNodesPerDirection;

        map<Position, list<Node *>*> *boundaryNodes;
        
        unsigned TotalNodes();

        unsigned MeshDimensions();
        
        Node *node(unsigned i);

        Node *node(unsigned i, unsigned j);

        Node *node(unsigned i, unsigned j, unsigned k);
        
        Node *nodeFromID(unsigned ID);
    
    private:
        Array<Node *> *_nodesMatrix;
        
        map<Position, list<Node*>*> *CreateBoundaries();

        map<Position, list<Node*>*> *Create1DBoundaries();

        map<Position, list<Node*>*> *Create2DBoundaries();

        map<Position, list<Node*>*> *Create3DBoundaries();
    };
}