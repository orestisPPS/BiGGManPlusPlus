//
// Created by hal9000 on 4/3/23.
//

#include "IsoparametricNodeGraph.h"

#include <utility>

namespace Discretization {
    
    IsoParametricNodeGraph::IsoParametricNodeGraph(Node* node, unsigned graphDepth, map<vector<double>, Node *> *nodeMap,
                                                   map<Direction, unsigned>& nodesPerDirection, bool includeDiagonalNeighbours) :
                            _node(node), _graphDepth(graphDepth), _nodesPerDirection(nodesPerDirection) {
        _nodeMap = nodeMap;
        nodeGraph = new map<Position, vector<Node*>>();
        _findINeighborhoodRecursively(includeDiagonalNeighbours);
    }
    
    map<Position, vector<Node*>>* IsoParametricNodeGraph::getNodeGraph() const {
        return nodeGraph;
    }
    
    map<Position, vector<Node*>>* IsoParametricNodeGraph::getNodeGraph(const map<Position, short unsigned>& customDepth) const {
        for (auto &position : customDepth) {
            auto currentPosition = position.first;
            if (nodeGraph->find(currentPosition) != nodeGraph->end()) {
                auto currentDepth = position.second;
                if (currentDepth < nodeGraph->at(currentPosition).size()) {
                    nodeGraph->at(currentPosition).erase(nodeGraph->at(currentPosition).begin() + currentDepth, nodeGraph->at(currentPosition).end());
                }
            }
        }
        return nodeGraph;
    }
    
    map<Position, vector<vector<DegreeOfFreedom*>*>>* IsoParametricNodeGraph::getAllDOFGraph() const {
        auto dofGraph = new map<Position, vector<vector<DegreeOfFreedom*>*>>();
        for (auto &position : *nodeGraph) {
            dofGraph->insert({position.first, vector<vector<DegreeOfFreedom*>*>()});
            for (auto &node : position.second) {
                dofGraph->at(position.first).push_back(node->degreesOfFreedom);
            }
        }
        return dofGraph;
    }
    
    map<Position, vector<DegreeOfFreedom*>>* IsoParametricNodeGraph::getSpecificDOFGraph(DOFType dofType) const {
        auto dofGraph = new map<Position, vector<DegreeOfFreedom*>>();
        for (auto &position : *nodeGraph) {
            dofGraph->insert({position.first, vector<DegreeOfFreedom*>()});
            for (auto &node : position.second) {
                dofGraph->at(position.first).push_back(node->getDegreeOfFreedomPtr(dofType));
            }
        }
        return dofGraph;
    }
    
    //BUGGY
    map<Position, vector<DegreeOfFreedom*>>* IsoParametricNodeGraph::getSpecificDOFGraph(DOFType dofType, ConstraintType constraint) const {
        auto dofGraph = new map<Position, vector<DegreeOfFreedom*>>();
        for (auto &position : *nodeGraph) {
            for (auto &node : position.second) {
                if (node->getDegreeOfFreedomPtr(dofType)->id->constraintType() == constraint)
                    dofGraph->at(position.first).push_back(node->getDegreeOfFreedomPtr(dofType));
            }
        }
        for (auto& emptyPosition : *dofGraph) {
            if (emptyPosition.second.empty()) {
                dofGraph->erase(emptyPosition.first);
            }
        }
        
        return dofGraph;
    }
    
    void IsoParametricNodeGraph::_findINeighborhoodRecursively(bool includeDiagonalNeighbours) {
        auto nodeCoords = _node->coordinates.positionVector(Parametric);
        if (nodeCoords.size() == 2) {
            nodeCoords.push_back(0.0);
        }
        else if (nodeCoords.size() == 1) {
            nodeCoords.push_back(0.0);
            nodeCoords.push_back(0.0);
        }
        for (int i = 1; i < _graphDepth + 1; ++i) {
            if (includeDiagonalNeighbours)
                _findIDepthNeighborhood(i, nodeCoords);
            else
                _findIDepthNeighborhoodOnlyDiagonals(i, nodeCoords);
        }
    }
    
    void IsoParametricNodeGraph::_findIDepthNeighborhoodOnlyDiagonals(unsigned int depth, vector<double>& nodeCoords) {
        
        auto nn1 = _nodesPerDirection[One];
        auto nn2 = _nodesPerDirection[Two];
        auto nn3 = _nodesPerDirection[Three];
        
        auto id = *_node->id.global;
        auto k = id / (nn1 * nn2);
        
        auto parametricCoords = vector<double>{0, 0, 0};

        //Top
        parametricCoords = {nodeCoords[0], nodeCoords[1] + depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Top, parametricCoords, depth);
        //Left
        parametricCoords = {nodeCoords[0] - depth, nodeCoords[1], nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Left, parametricCoords, depth);
        //Right
        parametricCoords = {nodeCoords[0] + depth, nodeCoords[1], nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Right, parametricCoords, depth);
        //Bottom
        parametricCoords = {nodeCoords[0], nodeCoords[1] - depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Bottom, parametricCoords, depth);
        
        if (k > 0){
            //Front
            parametricCoords = {nodeCoords[0], nodeCoords[1], nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(Front, parametricCoords, depth);
        }
        if (k < nn3 - 1) {
            //Back
            parametricCoords = {nodeCoords[0], nodeCoords[1], nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(Back, parametricCoords, depth);
        }
        
    }


    void IsoParametricNodeGraph::_findIDepthNeighborhood(unsigned int depth, vector<double>& nodeCoords) {

        auto nn1 = _nodesPerDirection[One];
        auto nn2 = _nodesPerDirection[Two];
        auto nn3 = _nodesPerDirection[Three];

        auto id = *_node->id.global;
        auto k = id / (nn1 * nn2);

        auto parametricCoords = vector<double>{0, 0, 0};
        //TopLeft
        parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] + depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(LeftTop, parametricCoords, depth);
        //Top
        parametricCoords = {nodeCoords[0], nodeCoords[1] + depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Top, parametricCoords, depth);
        //TopRight
        parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] + depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(RightTop, parametricCoords, depth);
        //Left
        parametricCoords = {nodeCoords[0] - depth, nodeCoords[1], nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Left, parametricCoords, depth);
        //Right
        parametricCoords = {nodeCoords[0] + depth, nodeCoords[1], nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Right, parametricCoords, depth);
        //BottomRight
        parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] - depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(RightBottom, parametricCoords, depth);
        //Bottom
        parametricCoords = {nodeCoords[0], nodeCoords[1] - depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Bottom, parametricCoords, depth);
        //BottomLeft
        parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] - depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(LeftBottom, parametricCoords, depth);

        if (k > 0){
            //FrontTopLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] + depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(LeftTopFront, parametricCoords, depth);

            //FrontTop
            parametricCoords = {nodeCoords[0], nodeCoords[1] + depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(TopFront, parametricCoords, depth);
            //FrontTopRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] + depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(RightTopFront, parametricCoords, depth);
            //FrontRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1], nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(RightFront, parametricCoords, depth);
            //Front
            parametricCoords = {nodeCoords[0], nodeCoords[1], nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(Front, parametricCoords, depth);
            //FrontLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1], nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(LeftFront, parametricCoords, depth);
            //FrontBottomRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] - depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(RightBottomFront, parametricCoords, depth);
            //FrontBottom
            parametricCoords = {nodeCoords[0], nodeCoords[1] - depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(BottomFront, parametricCoords, depth);
            //FrontBottomLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] - depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(LeftBottomFront, parametricCoords, depth);
        }

        if (k < nn3 - 1) {
            //BackTopLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] + depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(LeftTopBack, parametricCoords, depth);
            //BackTop
            parametricCoords = {nodeCoords[0], nodeCoords[1] + depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(TopBack, parametricCoords, depth);
            //BackTopRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] + depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(RightTopBack, parametricCoords, depth);
            //BackLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1], nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(LeftBack, parametricCoords, depth);
            //Back
            parametricCoords = {nodeCoords[0], nodeCoords[1], nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(Back, parametricCoords, depth);
            //BackRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1], nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(RightBack, parametricCoords, depth);
            //BackBottomLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] - depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(LeftBottomBack, parametricCoords, depth);
            //BackBottom
            parametricCoords = {nodeCoords[0], nodeCoords[1] - depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(BottomBack, parametricCoords, depth);
            //BackBottomRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] - depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(RightBottomBack, parametricCoords, depth);
        }

    }
    
    
    
    void IsoParametricNodeGraph:: _addNeighbourNodeIfParametricCoordsExist(Position position,
                                                                           vector<double>& parametricCoords, unsigned currentDepth){
        if (_nodeMap->find(parametricCoords) != _nodeMap->end()){
            if (nodeGraph->find(position) == nodeGraph->end()){
                nodeGraph->insert(pair<Position, vector<Node*>>(position, vector<Node*>()));
            }
            if (nodeGraph->at(position).empty()){
                nodeGraph->at(position) = vector<Node*>();
            }
            nodeGraph->at(position).push_back(_nodeMap->at(parametricCoords));
        }
    }


} // Node