//
// Created by hal9000 on 12/21/22.
//

#include "NodeFactory.h"

namespace StructuredMeshGenerator{
    NodeFactory :: NodeFactory(map<Direction, int> *numberOfNodes){
        _nn1 = numberOfNodes->at(Direction::One);
        _nn2 = numberOfNodes->at(Direction::Two);
        _nn3 = numberOfNodes->at(Direction::Three);
        nodesMatrix = AllocateNodesMatrix();
        CreateNodes();
        AssignGlobalId();
    }
    
    Array<Node*> *NodeFactory :: AllocateNodesMatrix(){
        if (_nn2 == 0 && _nn3 == 0)
            return new Array<Node *>(_nn1);
        else if (_nn3 == 0)
            return new Array<Node *>(_nn1, _nn2);
        else
            return new Array<Node *>(_nn1, _nn2, _nn3);
    }

    void NodeFactory::CreateNodes() {
        if ((_nn2 == 0 && _nn3 == 0) || (_nn1 == 0 && _nn2) || (_nn1 == 0 && _nn3))
            Create1DBoundaryNodes();
        else if (_nn1 == 0 && (_nn2 > 0 && _nn3 > 0) ||
                (_nn2 == 0 && (_nn1 > 0 && _nn3 > 0) ||
                (_nn3 == 0 && (_nn1 > 0 && _nn2 > 0))))
            Create2DBoundaryNodes();
        else
            Create3DBoundaryNodes();
    }

    void NodeFactory::Create1DBoundaryNodes() {
                
        auto leftNode = AllocateBoundaryNode(0);
        nodesMatrix->populateElement(0, leftNode);
        
        if (_nn2 == 0 && _nn3 == 0){
            auto rightNode = AllocateBoundaryNode(_nn1 - 1);
            nodesMatrix->populateElement(_nn1 - 1, rightNode);
        }
        else if (_nn1 == 0 && _nn2 == 0){
            auto rightNode = AllocateBoundaryNode(_nn3 - 1);
            nodesMatrix->populateElement(_nn3 - 1, rightNode);
        }
        else if (_nn1 == 0 && _nn3 == 0){
            auto rightNode = AllocateBoundaryNode(_nn2 - 1);
            nodesMatrix->populateElement(_nn2 - 1, rightNode);
        }
        else
            throw "Two or more directions have non zero number of nodes.";
    }
    
    void NodeFactory::Create2DBoundaryNodes() {
        if (_nn3 == 0){
            auto boundaryId = 0;
            //Bottom boundary nodes.
            for (int i = 0; i < _nn1 ; ++i) {
                nodesMatrix->populateElement(i, 0,  AllocateBoundaryNode(boundaryId));
                boundaryId++;
            }
            //Right boundary nodes.
            for (int i = 1; i < _nn2 ; ++i) {
                nodesMatrix->populateElement(_nn1 - 1, i, AllocateBoundaryNode(boundaryId));
                boundaryId++;
            }
            //Top boundary nodes.
            for (int i = _nn1 - 2; i >= 0 ; --i) {
                nodesMatrix->populateElement(i, _nn2 - 1, AllocateBoundaryNode(boundaryId));
                boundaryId++;
            }
            //Left boundary nodes.
            for (int i = _nn2 - 2; i >= 1 ; --i) {
                nodesMatrix->populateElement(0, i, AllocateBoundaryNode(boundaryId));
                boundaryId++;
            }
        }
        
        else if (_nn2 == 0){
            auto boundaryId = 0;
            //Bottom boundary nodes.
            for (int i = 0; i < _nn1 ; ++i) {
                nodesMatrix->populateElement(i, 0,  AllocateBoundaryNode(boundaryId));
                boundaryId++;
            }
            //Right boundary nodes.
            for (int i = 1; i < _nn3 ; ++i) {
                nodesMatrix->populateElement(_nn1 - 1, i, AllocateBoundaryNode(boundaryId));
                boundaryId++;
            }
            //Top boundary nodes.
            for (int i = _nn1 - 2; i >= 0 ; --i) {
                nodesMatrix->populateElement(i, _nn3 - 1, AllocateBoundaryNode(boundaryId));
                boundaryId++;
            }
            //Left boundary nodes.
            for (int i = _nn3 - 2; i >= 1 ; --i) {
                nodesMatrix->populateElement(0, i, AllocateBoundaryNode(boundaryId));
                boundaryId++;
            }
        }
            
            else if (_nn1 == 0){
                auto boundaryId = 0;
                //Bottom boundary nodes.
                for (int i = 0; i < _nn2 ; ++i) {
                    nodesMatrix->populateElement(i, 0,  AllocateBoundaryNode(boundaryId));
                    boundaryId++;
                }
                //Right boundary nodes.
                for (int i = 1; i < _nn3 ; ++i) {
                    nodesMatrix->populateElement(_nn2 - 1, i, AllocateBoundaryNode(boundaryId));
                    boundaryId++;
                }
                //Top boundary nodes.
                for (int i = _nn2 - 2; i >= 0 ; --i) {
                    nodesMatrix->populateElement(i, _nn3 - 1, AllocateBoundaryNode(boundaryId));
                    boundaryId++;
                }
                //Left boundary nodes.
                for (int i = _nn3 - 2; i >= 1 ; --i) {
                    nodesMatrix->populateElement(0, i, AllocateBoundaryNode(boundaryId));
                    boundaryId++;
                }
            }
            else
                throw "Two or more directions have non zero number of nodes.";
            
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
    
    void NodeFactory::Create1DInternalNodes() {
        auto iMax = 0;
        if (_nn2 == 0 && _nn3 == 0)
            iMax = _nn1;
        else if (_nn1 == 0 && _nn3 == 0)
            iMax = _nn2;
        else if (_nn1 == 0 && _nn2 == 0)
            iMax = _nn3;
        else
            throw "Two or more directions have non zero number of nodes.";
        auto internalId = 0;
        for (int i = 1; i < iMax - 1; i++){
            nodesMatrix->populateElement(i, AllocateInternalNode(internalId));
            internalId++;
        }
    }
    
    void NodeFactory::Create2DInternalNodes() {
        auto internalId = 0;
        for (int i = 1; i < _nn1 - 1; i++){
            for (int j = 1; j < _nn3 - 1; j++){
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
        
    }





} // StructuredMeshGenerator