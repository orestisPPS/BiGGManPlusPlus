//
// Created by hal9000 on 4/3/23.
//

#include "IsoparametricNodeGraph.h"

#include <utility>
#include <algorithm>

namespace Discretization {

    IsoParametricNodeGraph::IsoParametricNodeGraph(Node *node, unsigned graphDepth,
                                                   map<vector<double>, Node *> *nodeMap,
                                                   map<Direction, unsigned> &nodesPerDirection,
                                                   bool includeDiagonalNeighbours) :
            _node(node), _graphDepth(graphDepth), _nodesPerDirection(nodesPerDirection) {
        _nodeMap = nodeMap;
        nodeGraph = new map<Position, vector<Node *>>();
        _findINeighborhoodRecursively(includeDiagonalNeighbours);
    }

    map<Position, vector<Node *>> *IsoParametricNodeGraph::getNodeGraph() {
        return nodeGraph;
    }

    map<Position, vector<Node *>> *
    IsoParametricNodeGraph::getNodeGraph(const map<Position, short unsigned> &customDepth) const {
        for (auto &position: customDepth) {
            auto currentPosition = position.first;
            if (nodeGraph->find(currentPosition) != nodeGraph->end()) {
                auto currentDepth = position.second;
                if (currentDepth < nodeGraph->at(currentPosition).size()) {
                    nodeGraph->at(currentPosition).erase(nodeGraph->at(currentPosition).begin() + currentDepth,
                                                         nodeGraph->at(currentPosition).end());
                }
            }
        }
        return nodeGraph;
    }

    map<Position, vector<vector<DegreeOfFreedom *> *>> *IsoParametricNodeGraph::getAllDOFGraph() const {
        auto dofGraph = new map<Position, vector<vector<DegreeOfFreedom *> *>>();
        for (auto &position: *nodeGraph) {
            dofGraph->insert({position.first, vector<vector<DegreeOfFreedom *> *>()});
            for (auto &node: position.second) {
                dofGraph->at(position.first).push_back(node->degreesOfFreedom);
            }
        }
        return dofGraph;
    }

    map<Position, vector<DegreeOfFreedom *>> *IsoParametricNodeGraph::getSpecificDOFGraph(DOFType dofType) const {
        auto dofGraph = new map<Position, vector<DegreeOfFreedom *>>();
        for (auto &position: *nodeGraph) {
            dofGraph->insert({position.first, vector<DegreeOfFreedom *>()});
            for (auto &node: position.second) {
                dofGraph->at(position.first).push_back(node->getDegreeOfFreedomPtr(dofType));
            }
        }
        return dofGraph;
    }

    //BUGGY
    map<Position, vector<DegreeOfFreedom *>> *
    IsoParametricNodeGraph::getSpecificDOFGraph(DOFType dofType, ConstraintType constraint) const {
        auto dofGraph = new map<Position, vector<DegreeOfFreedom *>>();
        for (auto &position: *nodeGraph) {
            for (auto &node: position.second) {
                if (node->getDegreeOfFreedomPtr(dofType)->id->constraintType() == constraint)
                    dofGraph->at(position.first).push_back(node->getDegreeOfFreedomPtr(dofType));
            }
        }
        for (auto &emptyPosition: *dofGraph) {
            if (emptyPosition.second.empty()) {
                dofGraph->erase(emptyPosition.first);
            }
        }
        return dofGraph;
    }

    map<Direction, vector<Node*>>* IsoParametricNodeGraph::getColinearNodes() const {
        auto coLinearNodes = new map<Direction, vector<Node *>>();
        auto direction = -1;
        auto colinearNodesVector = vector<Node *>();
        for (auto &position1: *nodeGraph) {
            auto n1 = normalUnitVectorsOfPositions.at(position1.first);
            for (auto &position2: *nodeGraph) {
                auto n2 = normalUnitVectorsOfPositions.at(position2.first);

                if (VectorOperations::dotProduct(n1, n2) == -1) {

                    if ((n1[0] + n2[0] == 0) && (n1[1] == n2[1]) && (n1[2] == n2[2])) {
                        colinearNodesVector = _mergeAndSortColinearNodes(position1.second, position2.second, _node);
                        direction = One;
                    } else if ((n1[0] == n2[0]) && (n1[1] + n2[1] == 0) && (n1[2] == n2[2])) {
                        colinearNodesVector = _mergeAndSortColinearNodes(position1.second, position2.second, _node);
                        direction = Two;
                    } else if ((n1[0] == n2[0]) && (n1[1] == n2[1]) && (n1[2] + n2[2] == 0)) {
                        colinearNodesVector = _mergeAndSortColinearNodes(position1.second, position2.second, _node);
                        direction = Three;
                    }
                    if (coLinearNodes->find(static_cast<const Direction>(direction)) == coLinearNodes->end()) {
                        coLinearNodes->insert({pair<Direction, vector<Node*>>(static_cast<const Direction>(direction), colinearNodesVector)});
                    }
                }
            }
        }
        return coLinearNodes;
    }

    vector<Node*> IsoParametricNodeGraph::_mergeAndSortColinearNodes(vector<Discretization::Node *> &nodesDirection1,
                                                                     vector<Discretization::Node *> &nodesDirection2,
                                                                     Discretization::Node *node) {
        auto mergedNodes = vector<Node *>( nodesDirection1.size() + nodesDirection2.size() + 1);
        std::merge(nodesDirection1.begin(), nodesDirection1.end(),
                   nodesDirection2.begin(), nodesDirection2.end(),
                   mergedNodes.begin());
        mergedNodes[nodesDirection1.size() + nodesDirection2.size()] = node;
        std::sort(mergedNodes.begin(), mergedNodes.end(), [](Node *a, Node *b) {
            auto coords1 = a->coordinates.positionVector(Parametric);
            auto coords2 = b->coordinates.positionVector(Parametric);
            return coords1 < coords2;
        });  /*;
        for (auto &node: mergedNodes) {
            node->printNode();
        }
        cout<<"-------------------------"<<endl;*/
        return mergedNodes;
    }
        

    
    vector<Node*> IsoParametricNodeGraph::getColinearNodes(Direction direction) const {
        auto coLinearNodes = getColinearNodes();
        auto nodes = coLinearNodes->at(direction);
        delete coLinearNodes;
        return nodes;
    }

    map<Direction, vector<vector<double>>> IsoParametricNodeGraph::
    getSameColinearNodalCoordinates(CoordinateType coordinateType) const {

        map<Direction, vector<vector<double>>> coLinearNodalCoordinates;
        auto coLinearNodes = getColinearNodes();
        auto numberOfDirections = coLinearNodes->size();
        for (auto &direction: *coLinearNodes) {
            auto directionI = direction.first;
            auto nodeVector = direction.second;
            coLinearNodalCoordinates.insert({directionI, vector<vector<double>>(numberOfDirections)});
            for (int i = 0; i < numberOfDirections; ++i) {
                coLinearNodalCoordinates.at(directionI)[i] = vector<double>(nodeVector.size());
                auto iNode = 0;
                for (auto &node: nodeVector) {
                    coLinearNodalCoordinates.at(directionI)[i][iNode] = node->coordinates.positionVector(coordinateType)[i];
                    iNode++;
                }

            }
        }
        delete coLinearNodes;
        return coLinearNodalCoordinates;
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