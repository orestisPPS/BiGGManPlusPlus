//
// Created by hal9000 on 12/21/22.
//
#include <vector>
#include "../Primitives/Matrix.h"
using namespace Primitives;
#include "../Discretization/Node/Node.h"
using namespace Discretization;
namespace StructuredMeshGenerator {

    class NodeFactory {
    public:
        NodeFactory(map<Direction, int> *nodesPerDirection);
        Primitives::Array<Node *> *nodesMatrix;
        
    private:
        //Number of nodes at direction One. (can be x, ξ, θ)
        int _nn1;
        
        //Number of nodes at direction Two. (can be y, η, r)
        int _nn2;
        
        //Number of nodes at direction Three. (can be z, ζ, φ)
        int _nn3;
        
        //Allocates memory for nodes matrix with respect to number of nodes in each direction.
        Array<Node*> *AllocateNodesMatrix();
                
        //Assigns global id to each node.
        void AssignGlobalId();
        
        //Creates boundary nodes.
        void CreateNodes();
        
        //Creates boundary nodes for 1D mesh (nnOne > 0, nnTwo = 0, nnThree = 0)
        //RELATIVE : left node has id 0, right node has id nnOne - 1
        void Create1DBoundaryNodes();
        
        //Creates boundary nodes for 2D mesh (nnOne > 0, nnTwo > 0, nnThree = 0),
        // with the following order in identification starting from (0,0):
        // <- <- <-
        // v      ^
        // v      ^
        // v      ^
        // ->->->->
        void Create2DBoundaryNodes();
        
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
        void Create1DInternalNodes();
        
        //Creates internal nodes for 2D mesh (nnOne > 0, nnTwo > 0, nnThree = 0).
        void Create2DInternalNodes();
        
        //Creates internal nodes for 3D mesh (nnOne > 0, nnTwo > 0, nnThree > 0).
        void Create3DInternalNodes();

        //Allocates memory for internal node.
        Node *AllocateInternalNode(int internalId);

    };
    


} // StructuredMeshGenerator
