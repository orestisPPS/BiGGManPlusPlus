//
// Created by hal9000 on 12/21/22.
//

#include "NodeFactory.h"

namespace StructuredMeshGenerator{
    
    NodeFactory :: NodeFactory(map<Direction, unsigned> &numberOfNodes){
        _nn1 = numberOfNodes.at(Direction::One);
        _nn2 = numberOfNodes.at(Direction::Two);
        _nn3 = numberOfNodes.at(Direction::Three);

        auto space = FindSpaceEntityType();
        CreateNodesArray(space);
    }
    
    SpaceEntityType NodeFactory::FindSpaceEntityType() const  {
        auto space = NullSpace;
        if (_nn2 == 1 && _nn3 == 1){
            space = Axis;
        } else if (_nn3 == 1){
            space = Plane;
        } else {
            space = Volume;
        }
        return space;
    }
    
    void NodeFactory :: CreateNodesArray(SpaceEntityType space){
        switch (space) {
            case SpaceEntityType::Axis:
                nodesMatrix = new Array<Node*>(_nn1, 1);
                Create1DBoundaryNodes(_nn1);
                Create1DInternalNodes(_nn1);
                Assign1DGlobalId();
                break;
            case SpaceEntityType::Plane:
                nodesMatrix = new Array<Node*>(_nn1, _nn2);
                Create2DBoundaryNodes(_nn1, _nn2);
                Create2DInternalNodes(_nn1, _nn2);
                Assign2DGlobalId();
                break;
            case SpaceEntityType::Volume:
                nodesMatrix = new Array<Node*>(_nn1, _nn2, _nn3);
                Create3DBoundaryNodes();
                Create3DInternalNodes();
                Assign3DGlobalId();
                break;
            default:
                throw runtime_error("Invalid space type");
        }
    }

    void NodeFactory::Create1DBoundaryNodes(unsigned position) const {
        int a = 1;
        Node *node = AllocateBoundaryNode(a);
        (*nodesMatrix)(0) = node;
        (*nodesMatrix)(a) = node;
        (*nodesMatrix)(0) = AllocateBoundaryNode(0);
        (*nodesMatrix)(position - 1) = AllocateBoundaryNode(position - 1);
    }
    
    void NodeFactory::Create2DBoundaryNodes(unsigned index1, unsigned index2) const {
        
        auto boundaryId = 0;
        //Bottom boundary nodes.
        for (auto i = 0; i < index1 ; ++i) {
            nodesMatrix->at(i, 0) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, 0) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //Right boundary nodes.
        for (auto i = 1; i < index2 ; ++i) {
            nodesMatrix->at(index1 - 1, i) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(index1 - 1, i) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //Top boundary nodes.
        for (int i = index1 - 2; i >= 0 ; --i) {
            nodesMatrix->at(i, index2 - 1) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, index2 - 1) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //Left boundary nodes.
        for (auto i = index2 - 2; i >= 1 ; --i) {
            nodesMatrix->at(0, i) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(0, i) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }         
    }
    
    void NodeFactory::Create3DBoundaryNodes() const {
        auto boundaryId = 0;
        //FrontBottom boundary nodes.
        for (int i = 0; i < _nn1 ; ++i) {
            nodesMatrix->at(i, 0, 0) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, 0, 0) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //FrontRight boundary nodes.
        for (int i = 1; i < _nn2 ; ++i) {
            nodesMatrix->at(_nn1 - 1, i, 0) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(_nn1 - 1, i, 0) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //FrontTop boundary nodes.
        for (int i = _nn1 - 2; i >= 0 ; --i) {
            nodesMatrix->at(i, _nn2 - 1, 0) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, _nn2 - 1, 0) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //FrontLeft boundary nodes.
        for (auto i = _nn2 - 2; i >= 1 ; --i) {
            nodesMatrix->at(0, i, 0) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(0, i, 0) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //BackBottom boundary nodes.
        for (int i = 0; i < _nn1 ; ++i) {
            nodesMatrix->at(i, 0, _nn3 - 1) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, 0, _nn3 - 1) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //BackRight boundary nodes.
        for (int i = 1; i < _nn2 ; ++i) {
            nodesMatrix->at(_nn1 - 1, i, _nn3 - 1) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(_nn1 - 1, i, _nn3 - 1) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //BackTop boundary nodes.
        for (int i = _nn1 - 2; i >= 0 ; --i) {
            nodesMatrix->at(i, _nn2 - 1, _nn3 - 1) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, _nn2 - 1, _nn3 - 1) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //BackLeft boundary nodes.
        for (auto i = _nn2 - 2; i >= 1 ; --i) {
            nodesMatrix->at(0, i, _nn3 - 1) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(0, i, _nn3 - 1) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //LeftBottom boundary nodes.
        for (int i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->at(0, 0, i) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(0, 0, i) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //LeftTop boundary nodes.
        for (int i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->at(0, _nn2 - 1, i) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(0, _nn2 - 1, i) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //RightBottom boundary nodes.
        for (int i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->at(_nn1 - 1, 0, i) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(_nn1 - 1, 0, i) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //RightTop boundary nodes.
        for (int i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->at(_nn1 - 1, _nn2 - 1, i) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(_nn1 - 1, _nn2 - 1, i) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //BottomFront boundary nodes.
        for (int i = 1; i < _nn1 - 1 ; ++i) {
            nodesMatrix->at(i, 0, 0) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, 0, 0) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //BottomBack boundary nodes.
        for (int i = 1; i < _nn1 - 1 ; ++i) {
            nodesMatrix->at(i, 0, _nn3 - 1) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, 0, _nn3 - 1) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //TopFront boundary nodes.
        for (int i = 1; i < _nn1 - 1 ; ++i) {
            nodesMatrix->at(i, _nn2 - 1, 0) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, _nn2 - 1, 0) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //TopBack boundary nodes.
        for (int i = 1; i < _nn1 - 1 ; ++i) {
            nodesMatrix->at(i, _nn2 - 1, _nn3 - 1) = AllocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, _nn2 - 1, _nn3 - 1) = AllocateBoundaryNode(boundaryId);
            boundaryId++;
        }
    }

    Node *NodeFactory::AllocateBoundaryNode(int boundaryId) {
        
        Node *node = new Node();
        *node->id.boundary = boundaryId;
        //cout << "Boundary node " << boundaryId << " created." << endl;
        return node;
    }
    
    void NodeFactory::Create1DInternalNodes(unsigned maxIndex) const {
        auto internalId = 0;
        for (int i = 1; i < maxIndex - 1; i++){
            (*nodesMatrix)(i) = AllocateInternalNode(internalId);
            internalId++;
        }
    }
    
    void NodeFactory::Create2DInternalNodes(unsigned index1, unsigned index2) const {
        auto internalId = 0;
        for (int j = 1; j < index2 - 1; j++){
            for (int i = 1; i < index1 - 1; i++){
                (*nodesMatrix)(i, j) = AllocateInternalNode(internalId);
                //cout << "Internal node " << internalId << " created." << endl;
                internalId++;
            }
        }
    }
    
    void NodeFactory::Create3DInternalNodes() const {
        auto internalId = 0;
        for (int i = 1; i < _nn3 - 1; i++){
            for (int j = 1; j < _nn2 - 1; j++){
                for (int k = 1; k < _nn1 - 1; k++){
                    nodesMatrix->at(i, j, k) = AllocateInternalNode(internalId);
                    //(*nodesMatrix)(i, j, k) = AllocateInternalNode(internalId);
                    internalId++;
                }
            }
        }
    }
    
    Node *NodeFactory::AllocateInternalNode(unsigned internalId) {
        Node *node = new Node();
        *node->id.internal = internalId;
        //cout << "Internal node value: " << *node->value->global << endl;
        return node;
    }
    
    void NodeFactory::Assign1DGlobalId() const {
        for (int i = 0; i < _nn1; i++){
            (*nodesMatrix)(i)->id.global = new unsigned(i);
        }
    }
    
    void NodeFactory::Assign2DGlobalId() const {
        auto id = 0;
        for (int j = 0; j < _nn2; j++){
            for (int i = 0; i < _nn1; i++){
                //(*nodesMatrix)(i, j)->value.global = new unsigned(i + j * _nn1);
                (*nodesMatrix)(i, j)->id.global = new unsigned(id);
                id++;
            }
        }
    }
    
    void NodeFactory::Assign3DGlobalId() const {
        auto id = 0;
        for (int k = 0; k < _nn3; k++){
            for (int j = 0; j < _nn2; j++){
                for (int i = 0; i < _nn1; i++){
                    //(*nodesMatrix)(i, j, k)->value.global = new unsigned(i + j * _nn1 + k * _nn1 * _nn2);
                    (*nodesMatrix)(i, j, k)->id.global = new unsigned(id);
                    id++;
                }
            }
        }
    }

} // StructuredMeshGenerator