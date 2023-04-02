/*
//
// Created by hal9000 on 4/1/23.
//

#include <algorithm>
#include "IsoParametricCurves.h"

namespace PositioningInSpace {

    IsoParametricCurves::IsoParametricCurves(map<Direction, map<double, Node *> *> *inputIsoParametricCurves) :
            isoParametricCurves(inputIsoParametricCurves) {}

    Node *IsoParametricCurves::getNodeAtParametricDirectionAndCoordinate(Direction direction, double coordinate) {
        return isoParametricCurves->at(direction)->at(coordinate);
    }

    unsigned IsoParametricCurves::getNodeIdAtParametricDirectionAndCoordinate(Direction direction, double coordinate) {
        return (*isoParametricCurves->at(direction)->at(coordinate)->id.global);
    }
    
    Node* IsoParametricCurves::getNodeWithParametricCoordinates(vector<double>* parametricCoordinates){
            
        if (isoParametricCurves->find(Direction::One) != isoParametricCurves->end() &&
            isoParametricCurves->at(Direction::One)->find(parametricCoordinates->at(0)) !=
            isoParametricCurves->at(Direction::One)->end()) {
            
            if (isoParametricCurves->find(Direction::Two) != isoParametricCurves->end() &&
                isoParametricCurves->at(Direction::Two)->find(parametricCoordinates->at(1)) !=
                isoParametricCurves->at(Direction::Two)->end()) {
                
                if (isoParametricCurves->find(Direction::Three) != isoParametricCurves->end() &&
                    isoParametricCurves->at(Direction::Three)->find(parametricCoordinates->at(2)) !=
                    isoParametricCurves->at(Direction::Three)->end()) {
                    
                    return isoParametricCurves->at(Direction::Three)->at(parametricCoordinates->at(2));
                }
                return isoParametricCurves->at(Direction::Two)->at(parametricCoordinates->at(1));
            }
            return isoParametricCurves->at(Direction::One)->at(parametricCoordinates->at(0));
        }
        return nullptr;
    }
    
    unsigned IsoParametricCurves::getNodeIdWithParametricCoordinates(vector<double>* parametricCoordinates){
        return (*getNodeWithParametricCoordinates(parametricCoordinates)->id.global);
    }

    vector<Node *> IsoParametricCurves::getIsoCurveNodes(Direction direction, double constantCoordinate) {
        auto isoCurveNodes = vector<Node *>();
        switch (direction) {
            case Direction::One:
                if (isoParametricCurves->find(Direction::Two) != isoParametricCurves->end() &&
                    isoParametricCurves->at(Direction::Two)->find(constantCoordinate) !=
                    isoParametricCurves->at(Direction::Two)->end()) {
                    isoCurveNodes.push_back(isoParametricCurves->at(Direction::Two)->at(constantCoordinate));
                }
                if (isoParametricCurves->find(Direction::Three) != isoParametricCurves->end() &&
                    isoParametricCurves->at(Direction::Three)->find(constantCoordinate) !=
                    isoParametricCurves->at(Direction::Three)->end()) {
                    isoCurveNodes.push_back(isoParametricCurves->at(Direction::Three)->at(constantCoordinate));
                }
                break;
            case Direction::Two:
                if (isoParametricCurves->find(Direction::One) != isoParametricCurves->end() &&
                    isoParametricCurves->at(Direction::One)->find(constantCoordinate) !=
                    isoParametricCurves->at(Direction::One)->end()) {
                    isoCurveNodes.push_back(isoParametricCurves->at(Direction::One)->at(constantCoordinate));
                }
                if (isoParametricCurves->find(Direction::Three) != isoParametricCurves->end() &&
                    isoParametricCurves->at(Direction::Three)->find(constantCoordinate) !=
                    isoParametricCurves->at(Direction::Three)->end()) {
                    isoCurveNodes.push_back(isoParametricCurves->at(Direction::Three)->at(constantCoordinate));
                }
                break;
            case Direction::Three:
                if (isoParametricCurves->find(Direction::One) != isoParametricCurves->end() &&
                    isoParametricCurves->at(Direction::One)->find(constantCoordinate) !=
                    isoParametricCurves->at(Direction::One)->end()) {
                    isoCurveNodes.push_back(isoParametricCurves->at(Direction::One)->at(constantCoordinate));
                }
                if (isoParametricCurves->find(Direction::Two) != isoParametricCurves->end() &&
                    isoParametricCurves->at(Direction::Two)->find(constantCoordinate) !=
                    isoParametricCurves->at(Direction::Two)->end()) {
                    isoCurveNodes.push_back(isoParametricCurves->at(Direction::Two)->at(constantCoordinate));
                }
                break;
            case Time:
                //throw argument expression
                throw std::invalid_argument("This is some Einstein Shit!");
        }
        return isoCurveNodes;
    }


    vector<unsigned> IsoParametricCurves::getIsoCurveNodeIds(Direction direction, double constantCoordinate) {
        auto isoCurveNodes = getIsoCurveNodes(direction, constantCoordinate);
        auto isoCurveIds = vector<unsigned>(isoCurveNodes.size());
        for (int (i) = 0; (i) < isoCurveIds.size() ; ++(i)) {
            isoCurveIds[i] = (*isoCurveNodes[i]->id.global);            
        }
        return isoCurveIds;
    }
    
    
    

}// PositioningInSpace*/
