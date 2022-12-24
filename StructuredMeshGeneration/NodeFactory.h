 //
// Created by hal9000 on 12/21/22.
//
#pragma once
#include <vector>
#include "../Primitives/Array.h"
using namespace Primitives;
#include "../Discretization/Node/Node.h"
using namespace Discretization;
namespace StructuredMeshGenerator {

    class NodeFactory {
    public:
        NodeFactory(map<Direction, int> *nodesPerDirection);
        NodeFactory(int nn1, int nn2, int nn3);
        Primitives::Array<Node *> *nodesMatrix;
        
    private:
        //Number of nodes at direction One. (can be x, ξ, θ)
        int _nn1;
        
        //Number of nodes at direction Two. (can be y, η, r)
        int _nn2;
        
        //Number of nodes at direction Three. (can be z, ζ, φ)
        int _nn3;
                        
        //Assigns global id to each node.
        void AssignGlobalId();
        
        //Creates boundary nodes.
        void CreateNodesArray();
        
        //Creates boundary nodes for 1D mesh (nnOne > 0, nnTwo = 0, nnThree = 0)
        //RELATIVE : left node has id 0, right node has id nnOne - 1
        void Create1DBoundaryNodes(int position);
        
        //Creates boundary nodes for 2D mesh (nnOne > 0, nnTwo > 0, nnThree = 0),
        // with the following order in identification starting from (0,0):
        // <- <- <-
        // v      ^
        // v      ^
        // v      ^
        // ->->->->
        void Create2DBoundaryNodes(int index1, int index2);
        
        //Creates boundary nodes for 3D mesh (nnOne > 0, nnTwo > 0, nnThree > 0).
        // with the following order in identification starting from (0,0,0):
        // frontBottom -> frontRight-> frontTop -> frontLeft -> backBottom -> backRight -> backTop -> backLeft->
        // bottomLeft->bottomRight->topLeft->topRight
        // <- <- <-
        // v      ^
        // v      ^
        // v      ^
        // ->->->->
        void Create3DBoundaryNodes();
        
        //Allocates memory for boundary node.
        Node *AllocateBoundaryNode(int boundaryId);
        
        //Creates internal nodes for 1D mesh (nnOne > 0, nnTwo = 0, nnThree = 0).
        void Create1DInternalNodes(int maxIndex);
        
        //Creates internal nodes for 2D mesh (nnOne > 0, nnTwo > 0, nnThree = 0).
        void Create2DInternalNodes(int index1, int index2);
        
        //Creates internal nodes for 3D mesh (nnOne > 0, nnTwo > 0, nnThree > 0).
        void Create3DInternalNodes();

        //Allocates memory for internal node.
        Node *AllocateInternalNode(int internalId);
        
    };
    


} // StructuredMeshGenerator
