//
// Created by hal9000 on 12/21/22.
//

#include "NodeFactory.h"

namespace StructuredMeshGenerator{
    
    NodeFactory :: NodeFactory(map<Direction, unsigned> &numberOfNodes, SpaceCharacteristics *spaceCharacteristics){
        _nn1 = numberOfNodes.at(Direction::One);
        _nn2 = numberOfNodes.at(Direction::Two);
        _nn3 = numberOfNodes.at(Direction::Three);
        CreateNodesArray();
        CreateNodesArray();
        AssignGlobalId();
    }
    
    void NodeFactory :: CreateNodesArray(SpaceCharacteristics &spaceCharacteristics){
        if (spaceCharacteristics.physicalSpace == PhysicalSpaceEntities::One_axis){
            nodesMatrix = new Array<Node*>(_nn1);
            Create1DBoundaryNodes(_nn1);
            Create1DInternalNodes(_nn1);
        }

        else if (spaceCharacteristics.physicalSpace == PhysicalSpaceEntities::Two_axis){
            nodesMatrix = new Array<Node*>(_nn2);
            Create1DBoundaryNodes(_nn2);
            Create1DInternalNodes(_nn2);
        }

        else if (spaceCharacteristics.physicalSpace == PhysicalSpaceEntities::Three_axis){
            nodesMatrix = new Array<Node*>(_nn3);
            Create1DBoundaryNodes(_nn3);
            Create1DInternalNodes(_nn3);
        }

        else if (spaceCharacteristics.physicalSpace == PhysicalSpaceEntities::OneTwo_plane){
            nodesMatrix = new Array<Node*>(_nn1, _nn2);
            Create2DBoundaryNodes(_nn1, _nn2);
            Create2DInternalNodes(_nn1, _nn2);
        }

        else if (spaceCharacteristics.physicalSpace == PhysicalSpaceEntities::TwoThree_plane){
            nodesMatrix = new Array<Node*>(_nn2, _nn3);
            Create2DBoundaryNodes(_nn2, _nn3);
            Create2DInternalNodes(_nn2, _nn3);
        }

        else if (spaceCharacteristics.physicalSpace == PhysicalSpaceEntities::OneThree_plane){
            nodesMatrix = new Array<Node*>(_nn1, _nn3);
            Create2DBoundaryNodes(_nn1, _nn3);
            Create2DInternalNodes(_nn1, _nn3);
        }
        
        else if (spaceCharacteristics.physicalSpace == PhysicalSpaceEntities::OneTwoThree_volume){
            nodesMatrix = new Array<Node*>(_nn1, _nn2, _nn3);
            Create3DBoundaryNodes();
            Create3DInternalNodes();
        }
    }

    void NodeFactory::Create1DBoundaryNodes(unsigned position) const {
        int a = 1;
        Node *node = AllocateBoundaryNode(a);
        nodesMatrix->populateElement(a, node);
        nodesMatrix->populateElement(0, AllocateBoundaryNode(0));
        nodesMatrix->populateElement(position - 1, AllocateBoundaryNode(position - 1));
    }
    
    void NodeFactory::Create2DBoundaryNodes(unsigned index1, unsigned index2) const {
        
        auto boundaryId = 0;
        //Bottom boundary nodes.
        for (auto i = 0; i < index1 ; ++i) {
            nodesMatrix->populateElement(i, 0,  AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //Right boundary nodes.
        for (auto i = 1; i < index2 ; ++i) {
            nodesMatrix->populateElement(index1 - 1, i, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //Top boundary nodes.
        for (int i = index1 - 2; i >= 0 ; --i) {
            nodesMatrix->populateElement(i, index2 - 1, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //Left boundary nodes.
        for (auto i = index2 - 2; i >= 1 ; --i) {
            nodesMatrix->populateElement(0, i, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }         
    }
    
    void NodeFactory::Create3DBoundaryNodes() const {
        auto boundaryId = 0;
        //FrontBottom boundary nodes.
        for (int i = 0; i < _nn1 ; ++i) {
            nodesMatrix->populateElement(i, 0, 0, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //FrontRight boundary nodes.
        for (int i = 1; i < _nn2 ; ++i) {
            nodesMatrix->populateElement(_nn1 - 1, i, 0, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //FrontTop boundary nodes.
        for (int i = _nn1 - 2; i >= 0 ; --i) {
            nodesMatrix->populateElement(i, _nn2 - 1, 0, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //FrontLeft boundary nodes.
        for (auto i = _nn2 - 2; i >= 1 ; --i) {
            nodesMatrix->populateElement(0, i, 0, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //BackBottom boundary nodes.
        for (auto i = 0; i < _nn1 ; ++i) {
            nodesMatrix->populateElement(i, 0, _nn3 - 1, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //BackRight boundary nodes.
        for (int i = 1; i < _nn2 ; ++i) {
            nodesMatrix->populateElement(_nn1 - 1, i, _nn3 - 1, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //BackTop boundary nodes.
        for (int i = _nn1 - 2; i >= 0 ; --i) {
            nodesMatrix->populateElement(i, _nn2 - 1, _nn3 - 1, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //BackLeft boundary nodes.
        for (auto i = _nn2 - 2; i >= 1 ; --i) {
            nodesMatrix->populateElement(0, i, _nn3 - 1, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //BottomLeft boundary nodes.
        for (auto i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->populateElement(0, 0, i, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //BottomRight boundary nodes.
        for (auto i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->populateElement(_nn1 - 1, 0, i, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //TopLeft boundary nodes.
        for (auto i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->populateElement(0, _nn2 - 1, i, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //TopRight boundary nodes.
        for (auto i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->populateElement(_nn1 - 1, _nn2 - 1, i, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
    }

    Node *NodeFactory::AllocateBoundaryNode(int boundaryId) {
        Node *node = new Node();
        *node->id->boundary = boundaryId;
        return node;
    }
    
    void NodeFactory::Create1DInternalNodes(unsigned maxIndex) const {
        auto internalId = 0;
        for (int i = 1; i < maxIndex - 1; i++){
            nodesMatrix->populateElement(i, AllocateInternalNode(internalId));
            internalId++;
        }
    }
    
    void NodeFactory::Create2DInternalNodes(unsigned index1, unsigned index2) const {
        auto internalId = 0;
        for (int i = 1; i < index1 - 1; i++){
            for (int j = 1; j < index2 - 1; j++){
                nodesMatrix->populateElement(j, i, AllocateInternalNode(internalId));
                //cout << "Internal node: " << *nodesMatrix->element(i,j)->id->global << endl;
                internalId++;
            }
        }
    }
    
    void NodeFactory::Create3DInternalNodes() const {
        auto internalId = 0;
        for (int i = 1; i < _nn1 - 1; i++){
            for (int j = 1; j < _nn2 - 1; j++){
                for (int k = 1; k < _nn3 - 1; k++){
                    nodesMatrix->populateElement(k, j, i, AllocateInternalNode(internalId));
                    internalId++;
                }
            }
        }
    }
    
    Node *NodeFactory::AllocateInternalNode(unsigned internalId) {
        Node *node = new Node();
        *node->id->internal = internalId;
        //cout << "Internal node id: " << *node->id->global << endl;
        return node;
    }
    
    void NodeFactory::AssignGlobalId() const {
        unsigned id = 0;
        for(int k = 0; k < _nn3; k++){
            for(int j = 0; j < _nn2; j++){
                for(int i = 0; i < _nn1; i++){
                    *nodesMatrix->element(i,j,k)->id->global = id;
                    id++;
                }
            }
        }
        
    }
} // StructuredMeshGenerator