//
// Created by hal9000 on 12/21/22.
//

#include "NodeFactory.h"

#include <utility>
#include "../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"

namespace StructuredMeshGenerator{
    
    NodeFactory :: NodeFactory(map<Direction, unsigned> &numberOfNodes, PhysicalSpaceEntity &space){
        _nn1 = numberOfNodes.at(Direction::One);
        _nn2 = numberOfNodes.at(Direction::Two);
        _nn3 = numberOfNodes.at(Direction::Three);
        CreateNodesArray(space);
        AssignGlobalId();
    }
    
    void NodeFactory :: CreateNodesArray(PhysicalSpaceEntity &space){
        if (space.type() == PhysicalSpaceEntities::One_axis){
            nodesMatrix = new Array<Node*>(_nn1);
            Create1DBoundaryNodes(_nn1, space);
            Create1DInternalNodes(_nn1, space);
        }

        else if (space.type() == PhysicalSpaceEntities::Two_axis){
            nodesMatrix = new Array<Node*>(_nn2);
            Create1DBoundaryNodes(_nn2, space);
            Create1DInternalNodes(_nn2, space);
        }

        else if (space.type() == PhysicalSpaceEntities::Three_axis){
            nodesMatrix = new Array<Node*>(_nn3);
            Create1DBoundaryNodes(_nn3, space);
            Create1DInternalNodes(_nn3, space);
        }

        else if (space.type() == PhysicalSpaceEntities::OneTwo_plane){
            nodesMatrix = new Array<Node*>(_nn1, _nn2);
            Create2DBoundaryNodes(_nn1, _nn2, space);
            Create2DInternalNodes(_nn1, _nn2, space);
        }

        else if (space.type() == PhysicalSpaceEntities::TwoThree_plane){
            nodesMatrix = new Array<Node*>(_nn2, _nn3);
            Create2DBoundaryNodes(_nn2, _nn3, space);
            Create2DInternalNodes(_nn2, _nn3, space);
        }

        else if (space.type() == PhysicalSpaceEntities::OneThree_plane){
            nodesMatrix = new Array<Node*>(_nn1, _nn3);
            Create2DBoundaryNodes(_nn1, _nn3, space);
            Create2DInternalNodes(_nn1, _nn3, space);
        }
        
        else if (space.type() == PhysicalSpaceEntities::OneTwoThree_volume){
            nodesMatrix = new Array<Node*>(_nn1, _nn2, _nn3);
            Create3DBoundaryNodes(space);
            Create3DInternalNodes(space);
        }
    }

    void NodeFactory::Create1DBoundaryNodes(unsigned position, PhysicalSpaceEntity &space) const {
        int a = 1;
        Node *node = AllocateBoundaryNode(a, space);
        (*nodesMatrix)(0) = node;
        (*nodesMatrix)(a) = node;
        (*nodesMatrix)(0) = AllocateBoundaryNode(0, space);
        (*nodesMatrix)(position - 1) = AllocateBoundaryNode(position - 1, space);
    }
    
    void NodeFactory::Create2DBoundaryNodes(unsigned index1, unsigned index2, PhysicalSpaceEntity &space) const {
        
        auto boundaryId = 0;
        //Bottom boundary nodes.
        for (auto i = 0; i < index1 ; ++i) {
            (*nodesMatrix)(i, 0) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }
        //Right boundary nodes.
        for (auto i = 1; i < index2 ; ++i) {
            (*nodesMatrix)(index1 - 1, i) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }
        //Top boundary nodes.
        for (int i = index1 - 2; i >= 0 ; --i) {
            (*nodesMatrix)(i, index2 - 1) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }
        //Left boundary nodes.
        for (auto i = index2 - 2; i >= 1 ; --i) {
            (*nodesMatrix)(0, i) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }         
    }
    
    void NodeFactory::Create3DBoundaryNodes(PhysicalSpaceEntity &space) const {
        auto boundaryId = 0;
        //FrontBottom boundary nodes.
        for (int i = 0; i < _nn1 ; ++i) {
            (*nodesMatrix)(i, 0, 0) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }
        //FrontRight boundary nodes.
        for (int i = 1; i < _nn2 ; ++i) {
            (*nodesMatrix)(_nn1 - 1, i, 0) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }
        //FrontTop boundary nodes.
        for (int i = _nn1 - 2; i >= 0 ; --i) {
            (*nodesMatrix)(i, _nn2 - 1, 0) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }
        //FrontLeft boundary nodes.
        for (auto i = _nn2 - 2; i >= 1 ; --i) {
            (*nodesMatrix)(0, i, 0) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }
        //BackBottom boundary nodes.
        for (auto i = 0; i < _nn1 ; ++i) {
            (*nodesMatrix)(i, 0, _nn3 - 1) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }
        //BackRight boundary nodes.
        for (int i = 1; i < _nn2 ; ++i) {
            (*nodesMatrix)(_nn1 - 1, i, _nn3 - 1) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }
        //BackTop boundary nodes.
        for (int i = _nn1 - 2; i >= 0 ; --i) {
            (*nodesMatrix)(i, _nn2 - 1, _nn3 - 1) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }
        //BackLeft boundary nodes.
        for (auto i = _nn2 - 2; i >= 1 ; --i) {
            (*nodesMatrix)(0, i, _nn3 - 1) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }
        //BottomLeft boundary nodes.
        for (auto i = 1; i < _nn3 - 1 ; ++i) {
            (*nodesMatrix)(0, 0, i) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }
        //BottomRight boundary nodes.
        for (auto i = 1; i < _nn3 - 1 ; ++i) {
            (*nodesMatrix)(_nn1 - 1, 0, i) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }
        //TopLeft boundary nodes.
        for (auto i = 1; i < _nn3 - 1 ; ++i) {
            (*nodesMatrix)(0, _nn2 - 1, i) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }
        //TopRight boundary nodes.
        for (auto i = 1; i < _nn3 - 1 ; ++i) {
            (*nodesMatrix)(_nn1 - 1, _nn2 - 1, i) = AllocateBoundaryNode(boundaryId, space);
            boundaryId++;
        }
    }

    Node *NodeFactory::AllocateBoundaryNode(int boundaryId, PhysicalSpaceEntity &space) {
        Node *node = new Node(space);
        *node->id.boundary = boundaryId;
        return node;
    }
    
    void NodeFactory::Create1DInternalNodes(unsigned maxIndex, PhysicalSpaceEntity &space) const {
        auto internalId = 0;
        for (int i = 1; i < maxIndex - 1; i++){
            (*nodesMatrix)(i) = AllocateInternalNode(internalId, space);
            internalId++;
        }
    }
    
    void NodeFactory::Create2DInternalNodes(unsigned index1, unsigned index2, PhysicalSpaceEntity &space) const {
        auto internalId = 0;
        for (int i = 1; i < index1 - 1; i++){
            for (int j = 1; j < index2 - 1; j++){
                (*nodesMatrix)(i, j) = AllocateInternalNode(internalId, space);
                //cout << "Internal node: " << *nodesMatrix->element(i,j)->id->global << endl;
                internalId++;
            }
        }
    }
    
    void NodeFactory::Create3DInternalNodes(PhysicalSpaceEntity &space) const {
        auto internalId = 0;
        for (int i = 1; i < _nn1 - 1; i++){
            for (int j = 1; j < _nn2 - 1; j++){
                for (int k = 1; k < _nn3 - 1; k++){
                    (*nodesMatrix)(i, j, k) = AllocateInternalNode(internalId, space);
                    internalId++;
                }
            }
        }
    }
    
    Node *NodeFactory::AllocateInternalNode(unsigned internalId, PhysicalSpaceEntity &space) {
        Node *node = new Node(space);
        *node->id.internal = internalId;
        //cout << "Internal node id: " << *node->id->global << endl;
        return node;
    }
    
    void NodeFactory::AssignGlobalId() const {
        unsigned id = 0;
        for(int k = 0; k < _nn3; k++){
            for(int j = 0; j < _nn2; j++){
                for(int i = 0; i < _nn1; i++){
                    (*nodesMatrix)(i, j, k)->id.global = new unsigned(id);
                    id++;
                }
            }
        }
        
    }
} // StructuredMeshGenerator