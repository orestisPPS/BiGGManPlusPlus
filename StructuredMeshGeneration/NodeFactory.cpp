//
// Created by hal9000 on 12/21/22.
//

#include "NodeFactory.h"

namespace StructuredMeshGenerator{
    
        NodeFactory::NodeFactory(map<Direction, int> *nodesPerDirection) {
            this->nodesPerDirection = nodesPerDirection;
        }
    
        Node *NodeFactory::AllocateInternalNode(int internalId) {
            Node *node = new Node();
            node->id->internal = new int (internalId);
            return node;
        }
        
        Node *NodeFactory::AllocateBoundaryNode(int boundaryId) {
            Node *node = new Node();
            node->id->boundary = new int (boundaryId);
            return node;
        }

        void NodeFactory::AssignGlobalId() {
            int globalId = 0;
            for (auto &node : mesh->nodes) {
                node->globalId = globalId;
                globalId++;
            }
        }
    
        void NodeFactory::CreateBoundaryNodes() {
            switch (nodesPerDirection->size()) {
                case 1:
                    Create1DBoundaryNodes();
                    break;
                case 2:
                    Create2DBoundaryNodes();
                    break;
                case 3:
                    Create3DBoundaryNodes();
                    break;
                default:
                    break;
            }
        }
        
        void NodeFactory::CreateInternalNodes() {
            switch (nodesPerDirection->size()) {
                case 1:
                    Create1DInternalNodes();
                    break;
                case 2:
                    Create2DInternalNodes();
                    break;
                case 3:
                    Create3DInternalNodes();
                    break;
                default:
                    break;
            }
        }
    
        void NodeFactory::Create1DBoundaryNodes() {
            
        }
} // StructuredMeshGenerator