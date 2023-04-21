//
// Created by hal9000 on 12/21/22.
//

#include "NodeFactory.h"

namespace StructuredMeshGenerator{
    
    NodeFactory :: NodeFactory(map<Direction,  short unsigned> &numberOfNodes){
        _nn1 = numberOfNodes.at(Direction::One);
        _nn2 = numberOfNodes.at(Direction::Two);
        _nn3 = numberOfNodes.at(Direction::Three);

        auto space = findSpaceEntityType();
        createNodesArray(space);
    }
    
    SpaceEntityType NodeFactory::findSpaceEntityType() const  {
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
    
    void NodeFactory :: createNodesArray(SpaceEntityType space){
        switch (space) {
            case SpaceEntityType::Axis:
                nodesMatrix = new Array<Node*>(_nn1, 1);
                create1DBoundaryNodes(_nn1);
                create1DInternalNodes(_nn1);
                assign1DGlobalId();
                break;
            case SpaceEntityType::Plane:
                nodesMatrix = new Array<Node*>(_nn1, _nn2);
                create2DBoundaryNodes(_nn1, _nn2);
                create2DInternalNodes(_nn1, _nn2);
                assign2DGlobalId();
                break;
            case SpaceEntityType::Volume:
                nodesMatrix = new Array<Node*>(_nn1, _nn2, _nn3);
                create3DBoundaryNodes();
                create3DInternalNodes();
                Assign3DGlobalId();
                break;
            default:
                throw runtime_error("Invalid space type");
        }
    }

    void NodeFactory::create1DBoundaryNodes(unsigned position) const {
        int a = 1;
        Node *node = allocateBoundaryNode(a);
        (*nodesMatrix)(0) = node;
        (*nodesMatrix)(a) = node;
        (*nodesMatrix)(0) = allocateBoundaryNode(0);
        (*nodesMatrix)(position - 1) = allocateBoundaryNode(position - 1);
    }
    
    void NodeFactory::create2DBoundaryNodes(unsigned index1, unsigned index2) const {
        
        auto boundaryId = 0;
        //Bottom boundary nodes.
        for (auto i = 0; i < index1 ; ++i) {
            nodesMatrix->at(i, 0) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, 0) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //Right boundary nodes.
        for (auto i = 1; i < index2 ; ++i) {
            nodesMatrix->at(index1 - 1, i) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(index1 - 1, i) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //Top boundary nodes.
        for (int i = index1 - 2; i >= 0 ; --i) {
            nodesMatrix->at(i, index2 - 1) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, index2 - 1) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //Left boundary nodes.
        for (auto i = index2 - 2; i >= 1 ; --i) {
            nodesMatrix->at(0, i) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(0, i) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }         
    }
    
    void NodeFactory::create3DBoundaryNodes() const {
        auto boundaryId = 0;
        //FrontBottom boundary nodes.
        for (int i = 0; i < _nn1 ; ++i) {
            nodesMatrix->at(i, 0, 0) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, 0, 0) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //FrontRight boundary nodes.
        for (int i = 1; i < _nn2 ; ++i) {
            nodesMatrix->at(_nn1 - 1, i, 0) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(_nn1 - 1, i, 0) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //FrontTop boundary nodes.
        for (int i = _nn1 - 2; i >= 0 ; --i) {
            nodesMatrix->at(i, _nn2 - 1, 0) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, _nn2 - 1, 0) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //FrontLeft boundary nodes.
        for (auto i = _nn2 - 2; i >= 1 ; --i) {
            nodesMatrix->at(0, i, 0) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(0, i, 0) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //BackBottom boundary nodes.
        for (int i = 0; i < _nn1 ; ++i) {
            nodesMatrix->at(i, 0, _nn3 - 1) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, 0, _nn3 - 1) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //BackRight boundary nodes.
        for (int i = 1; i < _nn2 ; ++i) {
            nodesMatrix->at(_nn1 - 1, i, _nn3 - 1) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(_nn1 - 1, i, _nn3 - 1) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //BackTop boundary nodes.
        for (int i = _nn1 - 2; i >= 0 ; --i) {
            nodesMatrix->at(i, _nn2 - 1, _nn3 - 1) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, _nn2 - 1, _nn3 - 1) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //BackLeft boundary nodes.
        for (auto i = _nn2 - 2; i >= 1 ; --i) {
            nodesMatrix->at(0, i, _nn3 - 1) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(0, i, _nn3 - 1) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //LeftBottom boundary nodes.
        for (int i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->at(0, 0, i) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(0, 0, i) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //LeftTop boundary nodes.
        for (int i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->at(0, _nn2 - 1, i) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(0, _nn2 - 1, i) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //RightBottom boundary nodes.
        for (int i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->at(_nn1 - 1, 0, i) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(_nn1 - 1, 0, i) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //RightTop boundary nodes.
        for (int i = 1; i < _nn3 - 1 ; ++i) {
            nodesMatrix->at(_nn1 - 1, _nn2 - 1, i) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(_nn1 - 1, _nn2 - 1, i) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //BottomFront boundary nodes.
        for (int i = 1; i < _nn1 - 1 ; ++i) {
            nodesMatrix->at(i, 0, 0) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, 0, 0) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //BottomBack boundary nodes.
        for (int i = 1; i < _nn1 - 1 ; ++i) {
            nodesMatrix->at(i, 0, _nn3 - 1) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, 0, _nn3 - 1) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //TopFront boundary nodes.
        for (int i = 1; i < _nn1 - 1 ; ++i) {
            nodesMatrix->at(i, _nn2 - 1, 0) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, _nn2 - 1, 0) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
        //TopBack boundary nodes.
        for (int i = 1; i < _nn1 - 1 ; ++i) {
            nodesMatrix->at(i, _nn2 - 1, _nn3 - 1) = allocateBoundaryNode(boundaryId);
            //(*nodesMatrix)(i, _nn2 - 1, _nn3 - 1) = allocateBoundaryNode(boundaryId);
            boundaryId++;
        }
    }

    Node *NodeFactory::allocateBoundaryNode(int boundaryId) {
        
        Node *node = new Node();
        *node->id.boundary = boundaryId;
        //cout << "Boundary node " << boundaryId << " created." << endl;
        return node;
    }
    
    void NodeFactory::create1DInternalNodes(unsigned maxIndex) const {
        auto internalId = 0;
        for (int i = 1; i < maxIndex - 1; i++){
            (*nodesMatrix)(i) = allocateInternalNode(internalId);
            internalId++;
        }
    }
    
    void NodeFactory::create2DInternalNodes(unsigned index1, unsigned index2) const {
        auto internalId = 0;
        for (int j = 1; j < index2 - 1; j++){
            for (int i = 1; i < index1 - 1; i++){
                (*nodesMatrix)(i, j) = allocateInternalNode(internalId);
                //cout << "Internal node " << internalId << " created." << endl;
                internalId++;
            }
        }
    }
    
    void NodeFactory::create3DInternalNodes() const {
        auto internalId = 0;
        for (int i = 1; i < _nn3 - 1; i++){
            for (int j = 1; j < _nn2 - 1; j++){
                for (int k = 1; k < _nn1 - 1; k++){
                    nodesMatrix->at(i, j, k) = allocateInternalNode(internalId);
                    //(*nodesMatrix)(i, j, k) = allocateInternalNode(internalId);
                    internalId++;
                }
            }
        }
    }
    
    Node *NodeFactory::allocateInternalNode(unsigned internalId) {
        Node *node = new Node();
        *node->id.internal = internalId;
        //cout << "Internal node value: " << *node->value->global << endl;
        return node;
    }
    
    void NodeFactory::assign1DGlobalId() const {
        for (int i = 0; i < _nn1; i++){
            (*nodesMatrix)(i)->id.global = new unsigned(i);
        }
    }
    
    void NodeFactory::assign2DGlobalId() const {
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