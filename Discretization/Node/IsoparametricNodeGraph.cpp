//
// Created by hal9000 on 4/3/23.
//

#include "IsoparametricNodeGraph.h"

#include <utility>
#include <algorithm>

namespace Discretization {
    
    IsoParametricNodeGraph::IsoParametricNodeGraph(Node* node, unsigned graphDepth, map<vector<double>, Node *> *nodeMap,
                                                   map<Direction, unsigned>& nodesPerDirection, bool includeDiagonalNeighbours) :
                            _node(node), _graphDepth(graphDepth), _nodesPerDirection(nodesPerDirection) {
        _nodeMap = nodeMap;
        nodeGraph = new map<Position, vector<Node*>>();
        _findINeighborhoodRecursively(includeDiagonalNeighbours);
    }
    
    map<Position, vector<Node*>>* IsoParametricNodeGraph::getNodeGraph() {
        return nodeGraph;
    }
    
    map<Position, vector<Node*>>* IsoParametricNodeGraph::getNodeGraph(const map<Position, short unsigned>& customDepth) {
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
    //TODO Debug this
    map<Direction, vector<Node*>>* IsoParametricNodeGraph::getColinearNodes() const{
         auto coLinearNodes = new map<Direction, vector<Node*>>();

        for (auto &position1 : *nodeGraph) {
            for (auto &position2: *nodeGraph) {
                auto n1 = normalUnitVectorsOfPositions.at(position1.first);
                auto n2 = normalUnitVectorsOfPositions.at(position2.first);
                if (VectorOperations::dotProduct(n1, n2) == -1) {
                    if (n1[0] + n2[0] == 0 && n1[0] != 0){
                        auto mergedNodes = vector<Node*>(position1.second.size() + position2.second.size() + 1);
                        std::merge(position1.second.begin(), position1.second.end(),
                                   position2.second.begin(), position2.second.end(),
                                   mergedNodes.begin());
                        coLinearNodes->insert(pair<Direction, vector<Node*>>(One, mergedNodes));
                        coLinearNodes->at(One)[position1.second.size() + position2.second.size()] = _node;
                    }
                    else if (n1[1] + n2[1] == 0 && n1[1] != 0){
                        auto mergedNodes = vector<Node*>(position1.second.size() + position2.second.size() + 1);
                        std::merge(position1.second.begin(), position1.second.end(),
                                   position2.second.begin(), position2.second.end(),
                                   mergedNodes.begin());
                        coLinearNodes->insert(pair<Direction, vector<Node*>>(Two, mergedNodes));
                        coLinearNodes->at(Two)[position1.second.size() + position2.second.size()] = _node;
                    }
                    else if (n1[2] + n2[2] == 0 && n1[2] != 0){
                        auto mergedNodes = vector<Node*>(position1.second.size() + position2.second.size() + 1);
                        std::merge(position1.second.begin(), position1.second.end(),
                                   position2.second.begin(), position2.second.end(),
                                   mergedNodes.begin());
                        coLinearNodes->insert(pair<Direction, vector<Node*>>(Three, mergedNodes));
                        coLinearNodes->at(Three)[position1.second.size() + position2.second.size()] = _node;
                    }
                    
                }
            }
        }
        
        auto i = 0;
        for (auto &direction : *coLinearNodes) {
            if (direction.second.empty())
                coLinearNodes->erase(direction.first);
            //coLinearNodes->at(direction.first).push_back(_node);
            std::sort(
                    coLinearNodes->at(direction.first).begin(), coLinearNodes->at(direction.first).end(), [direction](Node* a, Node* b) {
                        auto coords1 = a->coordinates.positionVector(Parametric);
                        auto coords2 = b->coordinates.positionVector(Parametric);
                        return coords1[spatialDirectionToUnsigned[direction.first]] < coords2[spatialDirectionToUnsigned[direction.first]];
                    });
            i++;
        }
        return coLinearNodes;
    }

    vector<Node*> IsoParametricNodeGraph::getColinearNodes(Direction direction) const {
        auto coLinearNodes = getColinearNodes();
        auto nodes = coLinearNodes->at(direction);
        delete coLinearNodes;
        return nodes;
    }
    
    map<Direction, vector<double>> IsoParametricNodeGraph::
    getColinearNodalCoordinate(CoordinateType coordinateType) const {
        map<Direction, vector<double>> coLinearNodalCoordinates;
        auto coLinearNodes = getColinearNodes();
        for(auto& direction : *coLinearNodes){
            coLinearNodalCoordinates.insert({direction.first, vector<double>(direction.second.size())});
            auto iDirection = spatialDirectionToUnsigned[direction.first];
            auto iPoint = 0;
            for(auto& node : direction.second){
                coLinearNodalCoordinates.at(direction.first)[iPoint]=
                        node->coordinates.positionVector(coordinateType)[iDirection];
                iPoint++;
            }
        }
        delete coLinearNodes;
        return coLinearNodalCoordinates;
    }
    
    vector<double> IsoParametricNodeGraph::getColinearNodalCoordinate(CoordinateType coordinateType, Direction direction) const {
        auto coLinearNodalCoordinates = getColinearNodalCoordinate(coordinateType);
        auto coordinates = coLinearNodalCoordinates.at(direction);
        return coordinates;
    }
    
    
    map<Direction, vector<DegreeOfFreedom*>> IsoParametricNodeGraph::
    getColinearDOF(DOFType dofType) const {
        map<Direction, vector<DegreeOfFreedom*>> coLinearDOF;
        auto coLinearNodes = getColinearNodes();
        for(auto& direction : *coLinearNodes){
            coLinearDOF.insert({direction.first, vector<DegreeOfFreedom*>(direction.second.size())});
            auto iPoint = 0;
            for(auto& node : direction.second){
                coLinearDOF.at(direction.first)[iPoint] = node->getDegreeOfFreedomPtr(dofType);
                iPoint++;
            }
        }
        delete coLinearNodes;
        return coLinearDOF;
    }

    vector<double> IsoParametricNodeGraph::getColinearDOF(DOFType dofType, Direction direction) const {
        auto coLinearDOF = getColinearDOF(dofType).at(direction);;
        auto dofValues = vector<double>(coLinearDOF.size());
        auto iDof = 0;
        for (auto &dof : coLinearDOF) {
            dofValues[iDof] = dof->value();
            iDof++;
        }
        return dofValues;
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