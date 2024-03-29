 //
// Created by hal9000 on 12/21/22.
//
#pragma once
#include "../LinearAlgebra/Array/Array.h"
#include <utility>
#include "../Discretization/Node/Node.h"
#include "../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"

 using namespace Discretization;
using namespace LinearAlgebra;
namespace StructuredMeshGenerator {

    class NodeFactory {
    public:
        NodeFactory(map<Direction, unsigned> &nodesPerDirection);
                
        shared_ptr<Array<Node*>> nodesMatrix;
        
    private:
        //Number of nodes at direction One. (can be x, ξ, θ)
        unsigned _nn1;
        
        //Number of nodes at direction Two. (can be y, η, r)
        unsigned _nn2;
        
        //Number of nodes at direction Three. (can be z, ζ, φ)
        unsigned _nn3;
        
        //Returns the space type of the mesh.
        SpaceEntityType findSpaceEntityType() const;
        
        //Creates boundary nodes.
        void createNodesArray(SpaceEntityType space);
        
        //Creates boundary nodes for 1D mesh (nnOne > 0, nnTwo = 0, nnThree = 0)
        //RELATIVE : left node has value 0, right node has value nnOne - 1
        void create1DNodes(unsigned position) const;
        
        //Creates boundary nodes for 2D mesh (nnOne > 0, nnTwo > 0, nnThree = 0),
        // with the following order in identification starting from (0,0):
        // <- <- <-
        // v      ^
        // v      ^
        // v      ^
        // ->->->->
        void create2DNodes(unsigned index1, unsigned index2) const;
        
        //Creates boundary nodes for 3D mesh (nnOne > 0, nnTwo > 0, nnThree > 0).
        // with the following order in identification starting from (0,0,0):
        // frontBottom -> frontRight-> frontTop -> frontLeft -> backBottom -> backRight -> backTop -> backLeft->
        // bottomLeft->bottomRight->topLeft->topRight
        // <- <- <-
        // v      ^
        // v      ^
        // v      ^
        // ->->->->
        void create3DNodes() const;

    };
    


} // StructuredMeshGenerator
