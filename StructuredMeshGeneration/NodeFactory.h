//
// Created by hal9000 on 12/21/22.
//
#include <vector>
#include "../Primitives/Matrix.h"
#include "../Discretization/Node/Node.h"
using namespace Discretization;
namespace StructuredMeshGenerator {

    class NodeFactory {
    public:
        NodeFactory(map<Direction, int> *nodesPerDirection);
        
    private:
        map<Direction, int> *nodesPerDirection;
        Node *AllocateInternalNode(int internalId);
        Node *AllocateBoundaryNode(int boundaryId);
        void AssignGlobalId();
        void CreateBoundaryNodes();
        vector<Node> Create1DBoundaryNodes();
        Primitives::Matrix<Node,Node,Node> Create2DBoundaryNodes();
        void Create3DBoundaryNodes();
        void CreateInternalNodes();
        void Create1DInternalNodes();
        void Create2DInternalNodes();
        void Create3DInternalNodes();
        
        
    };
    


} // StructuredMeshGenerator
