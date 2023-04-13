//
// Created by hal9000 on 4/10/23.
//

#ifndef UNTITLED_GHOSTPSEUDOMESH_H
#define UNTITLED_GHOSTPSEUDOMESH_H

#include <map>
#include "../../Node/Node.h"
#include "../../../LinearAlgebra/Array.h"

using namespace Discretization;
using namespace LinearAlgebra;

namespace Discretization {

    class GhostPseudoMesh {
        
    public:
        
        GhostPseudoMesh(list<Node*>* ghostNodesList,
                        map<Direction, unsigned>* ghostNodesPerDirection,
                        map<vector<double>, Node*>* parametricCoordToNodeMap);
        
        ~GhostPseudoMesh();
        
        list<Node*>* ghostNodesList;

        map<Direction, unsigned>* ghostNodesPerDirection;
        
        //Contains the parametric coordinates of the nodes in the mesh (real and ghost)
        map<vector<double>, Node*> *parametricCoordToNodeMap;
    };

} // Discretization

#endif //UNTITLED_GHOSTPSEUDOMESH_H
