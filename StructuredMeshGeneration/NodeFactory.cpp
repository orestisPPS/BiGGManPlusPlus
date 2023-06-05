//
// Created by hal9000 on 12/21/22.
//

#include "NodeFactory.h"

namespace StructuredMeshGenerator{
    
    NodeFactory :: NodeFactory(map<Direction,  unsigned> &numberOfNodes){
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
        }
        else if (_nn3 == 1){
            space = Plane;
        } 
        else {
            space = Volume;
        }
        return space;
    }
    
    void NodeFactory :: createNodesArray(SpaceEntityType space){
        switch (space) {
            case SpaceEntityType::Axis:
                nodesMatrix = new Array<Node*>(_nn1, unsigned (1));
                create1DNodes(_nn1);

                break;
            case SpaceEntityType::Plane:
                nodesMatrix = new Array<Node*>(_nn1, _nn2);
                create2DNodes(_nn1, _nn2);
                break;
            case SpaceEntityType::Volume:
                nodesMatrix = new Array<Node*>(_nn1, _nn2, _nn3);
                create3DNodes();
                break;
            default:
                throw runtime_error("Invalid space type");
        }
    }

    void NodeFactory::create1DNodes(unsigned position) const {
        unsigned globalId = 0;
        for (int i = 0; i < position; ++i) {
            nodesMatrix->at(i) = new Node();
            *nodesMatrix->at(i)->id.global = globalId;
            globalId++;
        }
    }
    
    void NodeFactory::create2DNodes(unsigned nn1, unsigned nn2) const {
        unsigned globalId = 0;
        for (int j = 0; j < nn2; ++j) {
            for (int i = 0; i < nn1; ++i) {
                nodesMatrix->at(i, j) = new Node();
                *nodesMatrix->at(i, j)->id.global = globalId;
                globalId++;
            }
        }
    }
    
    void NodeFactory::create3DNodes() const {
        unsigned globalId = 0;
        for (int k = 0; k < _nn3; ++k) {
            for (int j = 0; j < _nn2; ++j) {
                for (int i = 0; i < _nn1; ++i) {
                    nodesMatrix->at(i, j, k) = new Node();
                    *nodesMatrix->at(i, j, k)->id.global = globalId;
                    globalId++;
                }
            }
        }
    }


} // StructuredMeshGenerator