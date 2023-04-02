//
// Created by hal9000 on 4/2/23.
//

#include "IsoParametricNeighbourFinder.h"

namespace Discretization {
    
    IsoParametricNeighbourFinder::IsoParametricNeighbourFinder(Mesh* mesh){
        _mesh = mesh;
        _parametricCoordinatesToNodeMap = _createParametricCoordinatesToNodeMap();
        maxMagnitude = Utility::Calculators::magnitude(vector<double>{
            static_cast<double>(mesh->numberOfNodesPerDirection[PositioningInSpace::One] - 1),
            static_cast<double>(mesh->numberOfNodesPerDirection[PositioningInSpace::Two] - 1),
            static_cast<double>(mesh->numberOfNodesPerDirection[PositioningInSpace::Three] - 1)});
        //auto hoodTest = getNeighbourNodes(12, 2);
        // auto dofHoodTest = getAllNeighbourDOF(12, 2);
    }
    
    IsoParametricNeighbourFinder::~IsoParametricNeighbourFinder(){
        cout << "IsoParametricNeighbourFinder destructor called" << endl;
    }
    
    
    map<vector<double>, Node*> IsoParametricNeighbourFinder::_createParametricCoordinatesToNodeMap() {
        auto parametricCoordMap = map<vector<double>, Node*>();
        for (auto &node : *_mesh->totalNodesVector) {
            auto parametricCoordinates = node->coordinates.positionVector(Parametric);
            if (parametricCoordinates.size() == 2){
                parametricCoordinates.push_back(0);
            }
            else if (parametricCoordinates.size() == 1){
                parametricCoordinates.push_back(0);
                parametricCoordinates.push_back(0);
            }
            parametricCoordMap.insert(pair<vector<double>, Node*> (parametricCoordinates, node));
        }
        return parametricCoordMap;
    }

    map<Position, vector<Node*>> IsoParametricNeighbourFinder::getNeighbourNodes(unsigned nodeId, unsigned depth){
        auto node = _mesh->totalNodesVector->at(nodeId);
        auto nodeCoords = node->coordinates.positionVector(Parametric);
        //write a functional programming function that calculates the length of a vector

        auto neighbourNodes = map<Position, vector<Node*>>();
        auto parametricCoords = vector<double>{0, 0, 0};
        for (int i = 1; i <= depth; ++i) {
            //TopLeft
            parametricCoords = {nodeCoords[0] - i, nodeCoords[1] + i, nodeCoords[2]};
            _addNodeToNeighboursIfParametricCoordsExists(TopLeft, parametricCoords, neighbourNodes);
            //Top
            parametricCoords = {nodeCoords[0], nodeCoords[1] + i, nodeCoords[2]};
            _addNodeToNeighboursIfParametricCoordsExists(Top, parametricCoords, neighbourNodes);
            //TopRight
            parametricCoords = {nodeCoords[0] + i, nodeCoords[1] + i, nodeCoords[2]};
            _addNodeToNeighboursIfParametricCoordsExists(TopRight, parametricCoords, neighbourNodes);
            //Left
            parametricCoords = {nodeCoords[0] - i, nodeCoords[1], nodeCoords[2]};
            _addNodeToNeighboursIfParametricCoordsExists(Left, parametricCoords, neighbourNodes);
            //Right
            parametricCoords = {nodeCoords[0] + i, nodeCoords[1], nodeCoords[2]};
            _addNodeToNeighboursIfParametricCoordsExists(Right, parametricCoords, neighbourNodes);
            //BottomRight
            parametricCoords = {nodeCoords[0] + i, nodeCoords[1] - i, nodeCoords[2]};
            _addNodeToNeighboursIfParametricCoordsExists(BottomRight, parametricCoords, neighbourNodes);
            //Bottom
            parametricCoords = {nodeCoords[0], nodeCoords[1] - i, nodeCoords[2]};
            _addNodeToNeighboursIfParametricCoordsExists(Bottom, parametricCoords, neighbourNodes);
            //BottomLeft
            parametricCoords = {nodeCoords[0] - i, nodeCoords[1] - i, nodeCoords[2]};   
            _addNodeToNeighboursIfParametricCoordsExists(BottomLeft, parametricCoords, neighbourNodes);

            //FrontTopLeft
            parametricCoords = {nodeCoords[0] - i, nodeCoords[1] + i, nodeCoords[2] + i};
            _addNodeToNeighboursIfParametricCoordsExists(FrontTopLeft, parametricCoords, neighbourNodes);
            
            //FrontTop
            parametricCoords = {nodeCoords[0], nodeCoords[1] + i, nodeCoords[2] + i};
            _addNodeToNeighboursIfParametricCoordsExists(FrontTop, parametricCoords, neighbourNodes);
            //FrontTopRight
            parametricCoords = {nodeCoords[0] + i, nodeCoords[1] + i, nodeCoords[2] + i};
            _addNodeToNeighboursIfParametricCoordsExists(FrontTopRight, parametricCoords, neighbourNodes);
            //FrontRight
            parametricCoords = {nodeCoords[0] + i, nodeCoords[1], nodeCoords[2] + i};
            _addNodeToNeighboursIfParametricCoordsExists(FrontRight, parametricCoords, neighbourNodes);
            //Front
            parametricCoords = {nodeCoords[0], nodeCoords[1], nodeCoords[2] + i};
            _addNodeToNeighboursIfParametricCoordsExists(Front, parametricCoords, neighbourNodes);
            //FrontLeft
            parametricCoords = {nodeCoords[0] - i, nodeCoords[1], nodeCoords[2] + i};
            _addNodeToNeighboursIfParametricCoordsExists(FrontLeft, parametricCoords, neighbourNodes);
            //FrontBottomRight
            parametricCoords = {nodeCoords[0] + i, nodeCoords[1] - i, nodeCoords[2] + i};
            _addNodeToNeighboursIfParametricCoordsExists(FrontBottomRight, parametricCoords, neighbourNodes);
            //FrontBottom
            parametricCoords = {nodeCoords[0], nodeCoords[1] - i, nodeCoords[2] + i};
            _addNodeToNeighboursIfParametricCoordsExists(FrontBottom, parametricCoords, neighbourNodes);
            //FrontBottomLeft
            parametricCoords = {nodeCoords[0] - i, nodeCoords[1] - i, nodeCoords[2] + i};
            _addNodeToNeighboursIfParametricCoordsExists(FrontBottomLeft, parametricCoords, neighbourNodes);
            //BackTopLeft
            parametricCoords = {nodeCoords[0] - i, nodeCoords[1] + i, nodeCoords[2] - i};
            _addNodeToNeighboursIfParametricCoordsExists(BackTopLeft, parametricCoords, neighbourNodes);
            //BackTop
            parametricCoords = {nodeCoords[0], nodeCoords[1] + i, nodeCoords[2] - i};
            _addNodeToNeighboursIfParametricCoordsExists(BackTop, parametricCoords, neighbourNodes);
            //BackTopRight
            parametricCoords = {nodeCoords[0] + i, nodeCoords[1] + i, nodeCoords[2] - i};
            _addNodeToNeighboursIfParametricCoordsExists(BackTopRight, parametricCoords, neighbourNodes);
            //BackLeft
            parametricCoords = {nodeCoords[0] - i, nodeCoords[1], nodeCoords[2] - i};
            _addNodeToNeighboursIfParametricCoordsExists(BackLeft, parametricCoords, neighbourNodes);
            //Back
            parametricCoords = {nodeCoords[0], nodeCoords[1], nodeCoords[2] - i};
            _addNodeToNeighboursIfParametricCoordsExists(Back, parametricCoords, neighbourNodes);
            //BackRight
            parametricCoords = {nodeCoords[0] + i, nodeCoords[1], nodeCoords[2] - i};
            _addNodeToNeighboursIfParametricCoordsExists(BackRight, parametricCoords, neighbourNodes);
            //BackBottomLeft
            parametricCoords = {nodeCoords[0] - i, nodeCoords[1] - i, nodeCoords[2] - i};
            _addNodeToNeighboursIfParametricCoordsExists(BackBottomLeft, parametricCoords, neighbourNodes);
            //BackBottom
            parametricCoords = {nodeCoords[0], nodeCoords[1] - i, nodeCoords[2] - i};
            _addNodeToNeighboursIfParametricCoordsExists(BackBottom, parametricCoords, neighbourNodes);
            //BackBottomRight
            parametricCoords = {nodeCoords[0] + i, nodeCoords[1] - i, nodeCoords[2] - i};
            _addNodeToNeighboursIfParametricCoordsExists(BackBottomRight, parametricCoords, neighbourNodes);
        }
        return neighbourNodes;
    }
    
    map<Position, vector<vector<DegreeOfFreedom*>>>
    IsoParametricNeighbourFinder::getAllNeighbourDOF(unsigned int nodeId, unsigned int depth) {
        auto neighbourNodes = this->getNeighbourNodes(nodeId, depth);
        map<Position, vector<vector<DegreeOfFreedom*>>> neighbourDOF;
        for (auto &neighbourNodePair : neighbourNodes){
            vector<vector<DegreeOfFreedom*>> neighbourDOFVector;
            for (auto &neighbourNode : neighbourNodePair.second){
                neighbourDOFVector.push_back(*neighbourNode->degreesOfFreedom);
            }
            neighbourDOF[neighbourNodePair.first] = neighbourDOFVector;
        }
        return neighbourDOF;
    }

    map<Position, vector<DegreeOfFreedom*>>
    IsoParametricNeighbourFinder::getSpecificNeighbourDOF(unsigned nodeId, DOFType dofType, unsigned depth) {
        auto allNeighbourDOF = getAllNeighbourDOF(nodeId, depth);
        map<Position, vector<DegreeOfFreedom*>> neighbourDOF;
        for (auto &neighbourDOFPair : allNeighbourDOF){
            for (auto &neighbourDOFVector : neighbourDOFPair.second){
                for (auto &neighbourDOFs : neighbourDOFVector){
                    if (neighbourDOFs->type() == dofType){
                        neighbourDOF[neighbourDOFPair.first] = neighbourDOFVector;
                    }
                }
            }
        }
        return neighbourDOF;
    }

    map<Position, vector<DegreeOfFreedom*>>
    IsoParametricNeighbourFinder::getSpecificNeighbourDOF(unsigned nodeId, DOFType dofType,
                                                          ConstraintType constraintType, unsigned depth) {
        auto allNeighbourDOF = getAllNeighbourDOF(nodeId, depth);
        map<Position, vector<DegreeOfFreedom*>> neighbourDOF;
        for (auto &neighbourDOFPair : allNeighbourDOF){
            for (auto &neighbourDOFVector : neighbourDOFPair.second){
                for (auto &neighbourDOFs : neighbourDOFVector){
                    if (neighbourDOFs->type() == dofType && neighbourDOFs->id->constraintType() == constraintType){
                        neighbourDOF[neighbourDOFPair.first] = neighbourDOFVector;
                    }
                }

            }
        }
        return neighbourDOF;
    }
    
    
    
    void IsoParametricNeighbourFinder::_addNodeToNeighboursIfParametricCoordsExists(Position position, vector<double>& parametricCoords,
                                                                                    map<Position,vector<Node*>> &neighbourNodesMap) {
        if (_parametricCoordinatesToNodeMap.find(parametricCoords) != _parametricCoordinatesToNodeMap.end()){
            neighbourNodesMap[position].push_back(_parametricCoordinatesToNodeMap[parametricCoords]);
        }
    }

} // Discretization