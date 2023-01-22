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
        Mesh(Array<Node *> *nodes, PhysicalSpaceEntity* space);
        
        ~Mesh();
        
        PhysicalSpaceEntity* space;
        
        //map<Direction, unsigned > *numberOfNodesPerDirection;
        map<Direction, unsigned > numberOfNodesPerDirection;

        map<Position, list<Node*>*> *boundaryNodes;

        const unsigned &totalNodes() const;

        const unsigned &dimensions() const;
        
        Node *node(unsigned i);

        Node *node(unsigned i, unsigned j);

        Node *node(unsigned i, unsigned j, unsigned k);
        
        Node *nodeFromID(unsigned ID);
        
        //Gets called by the mesh preprocessor to initiate space, numberOfNodesPerDirection, and dimensions
        void getSpatialProperties(map<Direction, unsigned> numberOfNodesPerDirection, PhysicalSpaceEntity space,
                                  unsigned dimensions, unsigned totalNodes);
    
    private:
        
        unsigned _dimensions;
        
        unsigned _totalNodes;
        
        bool _isInitialized;
        
        Array<Node *> *_nodesMatrix;
        
        map<Position, list<Node*>*> *CreateBoundaries();

        map<Position, list<Node*>*> *Create1DBoundaries();

        map<Position, list<Node*>*> *Create2DBoundaries();

        map<Position, list<Node*>*> *Create3DBoundaries();
    };
}