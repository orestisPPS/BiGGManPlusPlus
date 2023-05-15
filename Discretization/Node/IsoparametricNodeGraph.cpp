//
// Created by hal9000 on 4/3/23.
//

#include "IsoparametricNodeGraph.h"

#include <memory>


namespace Discretization {

    IsoParametricNodeGraph::IsoParametricNodeGraph(Node *node, unsigned graphDepth,
                                                   map<vector<double>, Node *> *nodeMap,
                                                   map<Direction, unsigned> &nodesPerDirection,
                                                   bool includeDiagonalNeighbours) :
            _node(node), _graphDepth(graphDepth), _nodesPerDirection(nodesPerDirection) {
        _nodeMap = nodeMap;
        nodeGraph = std::make_shared<map<Position, vector<Node*>>>();
        _findINeighborhoodRecursively(includeDiagonalNeighbours);
    }
    

    shared_ptr<map<Position, vector<Node*>>> IsoParametricNodeGraph::getNodeGraph() const {
        return shared_ptr<map<Position, vector<Node*>>>(nodeGraph);
    }

    shared_ptr<map<Position, vector<Node*>>>
    IsoParametricNodeGraph::getNodeGraph(const map<Position, short unsigned> &customDepth) const {
        shared_ptr<map<Position, vector<Node*>>> graph = shared_ptr<map<Position, vector<Node*>>>(this->nodeGraph);
        for (auto &position: customDepth) {
            auto currentPosition = position.first;
            if (graph->find(currentPosition) != graph->end()) {
                auto currentDepth = position.second;
                if (currentDepth < graph->at(currentPosition).size()) {
                    graph->at(currentPosition).erase(graph->at(currentPosition).begin() + currentDepth,
                                                     graph->at(currentPosition).end());
                }
            }
        }
        return graph;
    }

    unique_ptr<map<Position, vector<vector<DegreeOfFreedom*>*>>> IsoParametricNodeGraph::getAllDOFGraph() const {
        unique_ptr<map<Position, vector<vector<DegreeOfFreedom*>*>>> dofGraph = make_unique<map<Position, vector<vector<DegreeOfFreedom*>*>>>();
        for (auto &position: *nodeGraph) {
            dofGraph->insert({position.first, vector<vector<DegreeOfFreedom *> *>()});
            for (auto &node: position.second) {
                dofGraph->at(position.first).push_back(node->degreesOfFreedom);
            }
        }
        return dofGraph;
    }

    unique_ptr<map<Position, vector<vector<DegreeOfFreedom*>*>>> IsoParametricNodeGraph::
    getAllDOFGraph(unique_ptr<map<Position, vector<Node*>>>& customNodeGraph) {
        unique_ptr<map<Position, vector<vector<DegreeOfFreedom*>*>>> dofGraph = make_unique<map<Position, vector<vector<DegreeOfFreedom*>*>>>();
        for (auto &position: *customNodeGraph) {
            dofGraph->insert({position.first, vector<vector<DegreeOfFreedom *> *>()});
            for (auto &node: position.second) {
                dofGraph->at(position.first).push_back(node->degreesOfFreedom);
            }
        }
        return dofGraph;
    }
    
    

    unique_ptr<map<Position, vector<DegreeOfFreedom *>>> IsoParametricNodeGraph::getSpecificDOFGraph(DOFType dofType) const {
        unique_ptr<map<Position, vector<DegreeOfFreedom *>>> dofGraph = make_unique<map<Position, vector<DegreeOfFreedom *>>>();
        for (auto &position: *nodeGraph) {
            dofGraph->insert({position.first, vector<DegreeOfFreedom *>()});
            for (auto &node: position.second) {
                dofGraph->at(position.first).push_back(node->getDegreeOfFreedomPtr(dofType));
            }
        }
        return dofGraph;
    }
    
    unique_ptr<map<Position, vector<DegreeOfFreedom *>>> IsoParametricNodeGraph::
    getSpecificDOFGraph(DOFType dofType, unique_ptr<map<Position, vector<Node*>>>& customNodeGraph) {
        unique_ptr<map<Position, vector<DegreeOfFreedom *>>> dofGraph;
        for (auto &position: *customNodeGraph) {
            dofGraph->insert({position.first, vector<DegreeOfFreedom *>()});
            for (auto &node: position.second) {
                dofGraph->at(position.first).push_back(node->getDegreeOfFreedomPtr(dofType));
            }
        }
        return dofGraph;
    }

    
    unique_ptr<map<Position, vector<DegreeOfFreedom *>>> IsoParametricNodeGraph::
    getSpecificDOFGraph(DOFType dofType, ConstraintType constraint) const {
        unique_ptr<map<Position, vector<DegreeOfFreedom *>>> dofGraph;
        for (auto &position: *nodeGraph) {
            dofGraph->insert({position.first, vector<DegreeOfFreedom *>()});
            for (auto &node: position.second) {
                if (node->getDegreeOfFreedomPtr(dofType)->id->constraintType() == constraint)
                    dofGraph->at(position.first).push_back(node->getDegreeOfFreedomPtr(dofType));
            }
        }
        return dofGraph;
    }

    //BUGGY
    unique_ptr<map<Position, vector<DegreeOfFreedom *>>>
    IsoParametricNodeGraph::
    getSpecificDOFGraph(DOFType dofType, ConstraintType constraint, shared_ptr<map<Position, vector<Node*>>>& customNodeGraph) {
        unique_ptr<map<Position, vector<DegreeOfFreedom *>>> dofGraph = make_unique<map<Position, vector<DegreeOfFreedom *>>>();
        for (auto &position: *customNodeGraph) {
            dofGraph->insert({position.first, vector<DegreeOfFreedom *>()});
            for (auto &node: position.second) {
                if (node->getDegreeOfFreedomPtr(dofType)->id->constraintType() == constraint)
                    dofGraph->at(position.first).push_back(node->getDegreeOfFreedomPtr(dofType));
            }
        }
        return dofGraph;
    }

    unique_ptr<map<Direction, vector<Node*>>> IsoParametricNodeGraph::getColinearNodes() const {
        unique_ptr<map<Direction, vector<Node*>>> coLinearNodes = std::make_unique<map<Direction, vector<Node*>>>();
        
        auto direction = 10;
        auto colinearNodesVector = vector<Node *>();
        for (auto &position1: *nodeGraph) {
            auto n1 = normalUnitVectorsOfPositions.at(position1.first);
            for (auto &position2: *nodeGraph) {
                auto n2 = normalUnitVectorsOfPositions.at(position2.first);
                auto n1n2 = VectorOperations::dotProduct(n1, n2);
                if (n1n2 == -1 || n1n2 == 1) {

                    if ((n1[0] + n2[0] == 0) && (n1[1] == n2[1]) && (n1[2] == n2[2]) ||
                        (n1[0] + n2[0] == 2) && (n1[1] == n2[1]) && (n1[2] == n2[2])) {
                        colinearNodesVector = _mergeAndSortColinearNodes(position1.second, position2.second, _node);
                        coLinearNodes->insert(pair<Direction, vector<Node *>>(One, colinearNodesVector));
                    }
                    else if ((n1[0] == n2[0]) && (n1[1] + n2[1] == 0) && (n1[2] == n2[2]) ||
                               (n1[0] == n2[0]) && (n1[1] + n2[1] == 2) && (n1[2] == n2[2])) {
                        colinearNodesVector = _mergeAndSortColinearNodes(position1.second, position2.second, _node);
                        coLinearNodes->insert(pair<Direction, vector<Node *>>(Two, colinearNodesVector));
                    }
                    else if ((n1[0] == n2[0]) && (n1[1] == n2[1]) && (n1[2] + n2[2] == 0) ||
                               (n1[0] == n2[0]) && (n1[1] == n2[1]) && (n1[2] + n2[2] == 2)){
                        colinearNodesVector = _mergeAndSortColinearNodes(position1.second, position2.second, _node);\
                        coLinearNodes->insert(pair<Direction, vector<Node *>>(Three, colinearNodesVector));
                    }
                }
            }
        }
        return coLinearNodes;
    }
    
    unique_ptr<map<Direction, vector<Node*>>> IsoParametricNodeGraph::
    getColinearNodes(vector<Direction>& directions, shared_ptr<map<Position, vector<Node*>>>& customNodeGraph) const {
        unique_ptr<map<Direction, vector<Node*>>> coLinearNodes = make_unique<map<Direction, vector<Node*>>>();

        auto direction = 10;
        auto colinearNodesVector = vector<Node *>();
        for (auto &position1: *customNodeGraph) {
            auto n1 = normalUnitVectorsOfPositions.at(position1.first);
            for (auto &position2: *customNodeGraph) {
                auto n2 = normalUnitVectorsOfPositions.at(position2.first);

                if (VectorOperations::dotProduct(n1, n2) == -1) {

                    if ((n1[0] + n2[0] == 0) && (n1[1] == n2[1]) && (n1[2] == n2[2])) {
                        colinearNodesVector = _mergeAndSortColinearNodes(position1.second, position2.second, _node);
                        coLinearNodes->insert(pair<Direction, vector<Node*>>(One, colinearNodesVector));
                    } else if ((n1[0] == n2[0]) && (n1[1] + n2[1] == 0) && (n1[2] == n2[2])) {
                        colinearNodesVector = _mergeAndSortColinearNodes(position1.second, position2.second, _node);
                        coLinearNodes->insert(pair<Direction, vector<Node*>>(Two, colinearNodesVector));
                    } else if ((n1[0] == n2[0]) && (n1[1] == n2[1]) && (n1[2] + n2[2] == 0)) {
                        colinearNodesVector = _mergeAndSortColinearNodes(position1.second, position2.second, _node);\
                        coLinearNodes->insert(pair<Direction, vector<Node*>>(Three, colinearNodesVector));
                    }
                }
            }
        }
        return coLinearNodes;
    }}
    
    unique_ptr<map<Direction, map<vector<Position>, short unsigned>>> IsoParametricNodeGraph::
    getColinearPositionsAndPoints(vector<Direction>& availableDirections) const{
        auto positionsAtDirection = make_unique<map<Direction, map<vector<Position>, short unsigned>>>();
        auto availablePositions = vector<tuple<Position, unsigned>>();
        auto i = 0;
        for (auto &position: *nodeGraph) {
            availablePositions.emplace_back(position.first, position.second.size());
            i++;
        }
        for (auto &direction : availableDirections) {
            positionsAtDirection->insert(pair<Direction, map<vector<Position>, short unsigned>>(
                    direction, map<vector<Position>, short unsigned>()));
        }
        for (auto &tuple : availablePositions){
            auto position = get<0>(tuple);
            auto depth = get<1>(tuple);
            if (position == Left || position == Right)
                positionsAtDirection->at(One).insert(pair<vector<Position>, short unsigned>(vector<Position>{position}, depth));
            else if (position == Bottom || position == Top)
                positionsAtDirection->at(Two).insert(pair<vector<Position>, short unsigned>(vector<Position>{position}, depth));
            else if (position == Back || position == Front )
                positionsAtDirection->at(Three).insert(pair<vector<Position>, short unsigned>(vector<Position>{position}, depth));
        }
        
        for (auto &temp : *positionsAtDirection) {
            auto &positions = temp.second;
            auto sumPoints = vector<Position>(2);
            auto sumDepth = vector<short unsigned>(2);
            auto it = 0;
            for (auto &position : positions) {
                sumPoints[it] = position.first[0];
                sumDepth[it] = position.second;
                it++;
            }
            sort(sumPoints.begin(), sumPoints.end(), [](const Position &a, const Position &b) {
                return a > b;
            });
            
            positions.insert(pair<vector<Position>, short unsigned>(
                    sumPoints, *min_element(sumDepth.begin(), sumDepth.end())));
        }

        return positionsAtDirection;
    }

    unique_ptr<map<Direction, map<vector<Position>, short unsigned>>> IsoParametricNodeGraph::
    getColinearPositionsAndPoints(vector<Direction>& availableDirections, shared_ptr<map<Position, vector<Node*>>>& customNodeGraph) {
        
        auto positionsAtDirection = make_unique<map<Direction, map<vector<Position>, short unsigned>>>();
        auto availablePositions = vector<tuple<Position, unsigned>>();
        auto i = 0;
        for (auto &position: *customNodeGraph) {
            availablePositions.emplace_back(position.first, position.second.size());
            i++;
        }
        for (auto &direction : availableDirections) {
            positionsAtDirection->insert(pair<Direction, map<vector<Position>, short unsigned>>(
                    direction, map<vector<Position>, short unsigned>()));
        }
        for (auto &tuple : availablePositions){
            auto position = get<0>(tuple);
            auto depth = get<1>(tuple);
            if (position == Left || position == Right)
                positionsAtDirection->at(One).insert(pair<vector<Position>, short unsigned>(vector<Position>{position}, depth));
            else if (position == Bottom || position == Top)
                positionsAtDirection->at(Two).insert(pair<vector<Position>, short unsigned>(vector<Position>{position}, depth));
            else if (position == Back || position == Front )
                positionsAtDirection->at(Three).insert(pair<vector<Position>, short unsigned>(vector<Position>{position}, depth));
        }
        
        for (auto &temp : *positionsAtDirection) {
            auto &positions = temp.second;
            auto sumPoints = vector<Position>(2);
            auto sumDepth = vector<short unsigned>(2);
            auto it = 0;
            for (auto &position : positions) {
                sumPoints[it] = position.first[0];
                sumDepth[it] = position.second;
                it++;
            }
            sort(sumPoints.begin(), sumPoints.end(), [](const Position &a, const Position &b) {
                return a > b;
            });
            
            positions.insert(pair<vector<Position>, short unsigned>(
                    sumPoints, *min_element(sumDepth.begin(), sumDepth.end())));
        }

        return positionsAtDirection;
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
        });
        return mergedNodes;
    }
    
    vector<Node*> IsoParametricNodeGraph::getColinearNodes(Direction direction) const {
        auto coLinearNodes = getColinearNodes();
        auto nodes = coLinearNodes->at(direction);
        return nodes;
    }

    vector<Node*> IsoParametricNodeGraph::getColinearNodes(Direction direction, shared_ptr<map<Position, vector<Node*>>>& customNodeGraph) const {
        auto coLinearNodes = getColinearNodes();
        auto nodes = coLinearNodes->at(direction);
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
        return coLinearNodalCoordinates;
    }
    
    map<Direction, vector<vector<double>>> IsoParametricNodeGraph::
    getSameColinearNodalCoordinates(CoordinateType coordinateType, shared_ptr<map<Position, vector<Node*>>>& customNodeGraph) const {

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
        return coLinearDOF;
    }
    
    map<Direction, vector<DegreeOfFreedom*>> IsoParametricNodeGraph::
    getColinearDOF(DOFType dofType, shared_ptr<map<Position, vector<Node*>>>& customNodeGraph) const {
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
        return coLinearDOF;
    }

    vector<DegreeOfFreedom*> IsoParametricNodeGraph::getColinearDOF(DOFType dofType, Direction direction) const {
        auto coLinearDOF = getColinearDOF(dofType).at(direction);;
        auto dofValues = vector<DegreeOfFreedom*>(coLinearDOF.size());
        auto iDof = 0;
        for (auto &dof : coLinearDOF) {
            dofValues[iDof] = dof;
            iDof++;
        }
        return dofValues;
    }
    
    vector<DegreeOfFreedom*> IsoParametricNodeGraph::getColinearDOF(DOFType dofType, Direction direction, shared_ptr<map<Position, vector<Node *>>> &customNodeGraph) const {
        auto colinearDOF = getColinearDOF(dofType, customNodeGraph);
        auto coLinearDOFAtDirection = colinearDOF.at(direction);;
        auto dofValues = vector<DegreeOfFreedom*>(coLinearDOFAtDirection.size());
        auto iDof = 0;
        for (auto &dof : coLinearDOFAtDirection) {
            dofValues[iDof] = dof;
            iDof++;
        }
        return dofValues;
    }
    
    void IsoParametricNodeGraph::_findINeighborhoodRecursively(bool includeDiagonalNeighbours) {
        auto nodeCoords = _node->coordinates.positionVector3D(Parametric);

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
        _addNeighbourNodeIfParametricCoordsExist(Top, parametricCoords);
        //Left
        parametricCoords = {nodeCoords[0] - depth, nodeCoords[1], nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Left, parametricCoords);
        //Right
        parametricCoords = {nodeCoords[0] + depth, nodeCoords[1], nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Right, parametricCoords);
        //Bottom
        parametricCoords = {nodeCoords[0], nodeCoords[1] - depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Bottom, parametricCoords);
        
        if (k > 0){
            //Front
            parametricCoords = {nodeCoords[0], nodeCoords[1], nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(Front, parametricCoords);
        }
        if (k < nn3 - 1) {
            //Back
            parametricCoords = {nodeCoords[0], nodeCoords[1], nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(Back, parametricCoords);
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
        _addNeighbourNodeIfParametricCoordsExist(LeftTop, parametricCoords);
        //Top
        parametricCoords = {nodeCoords[0], nodeCoords[1] + depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Top, parametricCoords);
        //TopRight
        parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] + depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(RightTop, parametricCoords);
        //Left
        parametricCoords = {nodeCoords[0] - depth, nodeCoords[1], nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Left, parametricCoords);
        //Right
        parametricCoords = {nodeCoords[0] + depth, nodeCoords[1], nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Right, parametricCoords);
        //BottomRight
        parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] - depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(RightBottom, parametricCoords);
        //Bottom
        parametricCoords = {nodeCoords[0], nodeCoords[1] - depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Bottom, parametricCoords);
        //BottomLeft
        parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] - depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(LeftBottom, parametricCoords);

        if (k > 0){
            //FrontTopLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] + depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(LeftTopFront, parametricCoords);

            //FrontTop
            parametricCoords = {nodeCoords[0], nodeCoords[1] + depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(TopFront, parametricCoords);
            //FrontTopRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] + depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(RightTopFront, parametricCoords);
            //FrontRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1], nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(RightFront, parametricCoords);
            //Front
            parametricCoords = {nodeCoords[0], nodeCoords[1], nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(Front, parametricCoords);
            //FrontLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1], nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(LeftFront, parametricCoords);
            //FrontBottomRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] - depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(RightBottomFront, parametricCoords);
            //FrontBottom
            parametricCoords = {nodeCoords[0], nodeCoords[1] - depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(BottomFront, parametricCoords);
            //FrontBottomLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] - depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(LeftBottomFront, parametricCoords);
        }

        if (k < nn3 - 1) {
            //BackTopLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] + depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(LeftTopBack, parametricCoords);
            //BackTop
            parametricCoords = {nodeCoords[0], nodeCoords[1] + depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(TopBack, parametricCoords);
            //BackTopRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] + depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(RightTopBack, parametricCoords);
            //BackLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1], nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(LeftBack, parametricCoords);
            //Back
            parametricCoords = {nodeCoords[0], nodeCoords[1], nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(Back, parametricCoords);
            //BackRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1], nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(RightBack, parametricCoords);
            //BackBottomLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] - depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(LeftBottomBack, parametricCoords);
            //BackBottom
            parametricCoords = {nodeCoords[0], nodeCoords[1] - depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(BottomBack, parametricCoords);
            //BackBottomRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] - depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(RightBottomBack, parametricCoords);
        }

    }
    
    
    void IsoParametricNodeGraph::_addNeighbourNodeIfParametricCoordsExist(Position position,
                                                                          vector<double> &parametricCoords) {
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

