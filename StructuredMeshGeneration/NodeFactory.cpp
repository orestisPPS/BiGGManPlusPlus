//
// Created by hal9000 on 12/21/22.
//

#include "NodeFactory.h"

namespace StructuredMeshGenerator{
    NodeFactory :: NodeFactory(map<Direction, int> *numberOfNodes){
        _nn1 = numberOfNodes->at(Direction::One);
        _nn2 = numberOfNodes->at(Direction::Two);
        _nn3 = numberOfNodes->at(Direction::Three);
        CreateNodesArray();
        AssignGlobalId();
    }
    NodeFactory :: NodeFactory(int nn1, int nn2, int nn3){
        _nn1 = nn1;
        _nn2 = nn2;
        _nn3 = nn3;
        CreateNodesArray();
        AssignGlobalId();
    }
    
    void NodeFactory :: CreateNodesArray(){
        if (_nn2 == 0 && _nn3 == 0){
            nodesMatrix = new Array<Node*>(_nn1);
            Create1DBoundaryNodes(_nn1);
            Create1DInternalNodes(_nn1);
        }

        else if (_nn1 == 0 && _nn3 == 0){
            nodesMatrix = new Array<Node*>(_nn2);
            Create1DBoundaryNodes(_nn2);
            Create1DInternalNodes(_nn2);
        }

        else if (_nn1 == 0 && _nn2 == 0){
            nodesMatrix = new Array<Node*>(_nn3);
            Create1DBoundaryNodes(_nn3);
            Create1DInternalNodes(_nn3);
        }

        else if (_nn1 == 0){
            nodesMatrix = new Array<Node*>(_nn2, _nn3);
            Create2DBoundaryNodes(_nn2, _nn3);
            Create2DInternalNodes(_nn2, _nn3);
        }

        else if (_nn2 == 0){
            nodesMatrix = new Array<Node*>(_nn1, _nn3);
            Create2DBoundaryNodes(_nn1, _nn3);
            Create2DInternalNodes(_nn1, _nn3);
        }

        else if (_nn3 == 0){
            nodesMatrix = new Array<Node*>(_nn1, _nn2);
            Create2DBoundaryNodes(_nn1, _nn2);
            Create2DInternalNodes(_nn1, _nn2);
        }
        else{
            nodesMatrix = new Array<Node*>(_nn1, _nn2, _nn3);
            Create3DBoundaryNodes();
            Create3DInternalNodes();
        }
    }

    void NodeFactory::Create1DBoundaryNodes(int position) {
        nodesMatrix->populateElement(0, AllocateBoundaryNode(0));
        nodesMatrix->populateElement(position - 1, AllocateBoundaryNode(position - 1));
    }
    
    void NodeFactory::Create2DBoundaryNodes(int index1, int index2) {
        
        auto boundaryId = 0;
        //Bottom boundary nodes.
        for (int i = 0; i < index1 ; ++i) {
            nodesMatrix->populateElement(i, 0,  AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //Right boundary nodes.
        for (int i = 1; i < index2 ; ++i) {
            nodesMatrix->populateElement(index1 - 1, i, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //Top boundary nodes.
        for (int i = index1 - 2; i >= 0 ; --i) {
            nodesMatrix->populateElement(i, index2 - 1, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //Left boundary nodes.
        for (int i = index2 - 2; i >= 1 ; --i) {
            nodesMatrix->populateElement(0, i, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }         
    }
    
    void NodeFactory::Create3DBoundaryNodes() {
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
        for (int i = _nn2 - 2; i >= 1 ; --i) {
            nodesMatrix->populateElement(0, i, 0, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //BackBottom boundary nodes.
        for (int i = 0; i < _nn1 ; ++i) {
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
        for (int i = _nn2 - 2; i >= 1 ; --i) {
            nodesMatrix->populateElement(0, i, _nn3 - 1, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //BottomLeft boundary nodes.
        for (int i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->populateElement(0, 0, i, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //BottomRight boundary nodes.
        for (int i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->populateElement(_nn1 - 1, 0, i, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //TopLeft boundary nodes.
        for (int i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->populateElement(0, _nn2 - 1, i, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
        //TopRight boundary nodes.
        for (int i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->populateElement(_nn1 - 1, _nn2 - 1, i, AllocateBoundaryNode(boundaryId));
            boundaryId++;
        }
    }

    Node *NodeFactory::AllocateBoundaryNode(int boundaryId) {
        Node *node = new Node();
        *node->id->global = boundaryId;
        return node;
    }
    
    void NodeFactory::Create1DInternalNodes(int maxIndex) {
        auto internalId = 0;
        for (int i = 1; i < maxIndex - 1; i++){
            nodesMatrix->populateElement(i, AllocateInternalNode(internalId));
            internalId++;
        }
    }
    
    void NodeFactory::Create2DInternalNodes(int index1, int index2) {
        auto internalId = 0;
        for (int i = 1; i < index1 - 1; i++){
            for (int j = 1; j < index2 - 1; j++){
                nodesMatrix->populateElement(i, j, AllocateInternalNode(internalId));
                internalId++;
            }
        }
    }
    
    void NodeFactory::Create3DInternalNodes() {
        auto internalId = 0;
        for (int i = 1; i < _nn1 - 1; i++){
            for (int j = 1; j < _nn2 - 1; j++){
                for (int k = 1; k < _nn3 - 1; k++){
                    nodesMatrix->populateElement(i, j, k, AllocateInternalNode(internalId));
                    internalId++;
                }
            }
        }
    }
    
    Node *NodeFactory::AllocateInternalNode(int internalId) {
        Node *node = new Node();
        *node->id->global = internalId;
        return node;
    }
    
    void NodeFactory::AssignGlobalId() {
        auto id = 0;
        for(int k = 0; k < _nn3; k++){
            for(int j = 0; j < _nn2; j++){
                for(int i = 0; i < _nn1; i++){
                    *nodesMatrix->element(j, i, k)->id->global = id;
                    id++;
                }
            }
        }
        
    }





} // StructuredMeshGenerator