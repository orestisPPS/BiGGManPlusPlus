//
// Created by hal9000 on 4/3/23.
//

#ifndef UNTITLED_ISOPARAMETRICNODEHOODGRAPH_H
#define UNTITLED_ISOPARAMETRICNODEHOODGRAPH_H

#include "Node.h"
#include "../Mesh/Mesh.h"

using namespace Discretization;

namespace Discretization {

    class IsoParametricNodeHoodGraph {
        
    public:
        
        IsoParametricNodeHoodGraph(Node* node, unsigned graphDepth, Mesh* mesh, map<vector<double>, Node*> *nodeMap);

        map<Position,vector<Node*>>* neighborhood;
        
    private:

        Node* _node;
        
        unsigned int _graphDepth;
        
        Mesh* _mesh;

        map<vector<double>, Node*>* _nodeMap;
        
        
        void _findINeighborhoodRecursive();
        
        void _findIDepthNeighborhood(unsigned int depth, vector<double>& nodeCoords);

        void _addNeighbourNodeIfParametricCoordsExist(Position position, vector<double>& parametricCoords, unsigned depthSize);
    };

} // Node

#endif //UNTITLED_ISOPARAMETRICNODEHOODGRAPH_H
