//
// Created by hal9000 on 12/17/22.
//

#include "../Node/Node.h"
#include<vector>
#include "../../Primitives/Matrix.h"
using namespace Discretization;
using namespace Primitives;

namespace Discretization {

    class Mesh {
    public:
        Mesh(Array<Node *> *nodes, map<Direction, int> numberOfNodesPerDirection);

        Mesh(vector<Node *> *nodes, map<Direction, int> numberOfNodesPerDirection);

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
        
        vector<Node *> *_nodesVector;

        map<Position, list<Node*>*> *CreateBoundaries();

        map<Position, list<Node*>*> *Create1DBoundaries();

        map<Position, list<Node*>*> *Create2DBoundaries();

        map<Position, list<Node*>*> *Create3DBoundaries();
    };
}