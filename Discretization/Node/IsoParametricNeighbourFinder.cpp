//
// Created by hal9000 on 4/2/23.
//

#include "IsoParametricNeighbourFinder.h"

namespace Discretization {
    
    IsoParametricNeighbourFinder::IsoParametricNeighbourFinder(Mesh* mesh){
        _mesh = mesh;
        allPossibleNeighbourParametricCoords = _getAllPossibleNeighbourParametricCoords();
        _parametricCoordinatesToNodeMap = _createParametricCoordinatesToNodeMap();
        maximumParametricCoords = vector<double>(3);
        maximumParametricCoords[0] = _mesh-> numberOfNodesPerDirection[One] -  1;
        maximumParametricCoords[1] = _mesh-> numberOfNodesPerDirection[Two] -  1;
        maximumParametricCoords[2] = _mesh-> numberOfNodesPerDirection[Three] -  1;
    }
    
    
    map<Position, vector<double>> IsoParametricNeighbourFinder::_getAllPossibleNeighbourParametricCoords(){
        map<Position, vector<double>> allPossibleNeighbourParametricCoords;
        allPossibleNeighbourParametricCoords[TopLeft] = vector<double>{-1, 1, 0};
        allPossibleNeighbourParametricCoords[Top] = vector<double>{0, 1, 0};
        allPossibleNeighbourParametricCoords[TopRight] = vector<double>{1, 1, 0};
        allPossibleNeighbourParametricCoords[Right] = vector<double>{1, 0, 0};
        allPossibleNeighbourParametricCoords[BottomRight] = vector<double>{1, -1, 0};
        allPossibleNeighbourParametricCoords[Bottom] = vector<double>{0, -1, 0};
        allPossibleNeighbourParametricCoords[BottomLeft] = vector<double>{-1, -1, 0};
        allPossibleNeighbourParametricCoords[Left] = vector<double>{-1, 0, 0};
        allPossibleNeighbourParametricCoords[FrontTopLeft] = vector<double>{-1, 1, 1};
        allPossibleNeighbourParametricCoords[FrontTop] = vector<double>{0, 1, 1};
        allPossibleNeighbourParametricCoords[FrontTopRight] = vector<double>{1, 1, 1};
        allPossibleNeighbourParametricCoords[FrontRight] = vector<double>{1, 0, 1};
        allPossibleNeighbourParametricCoords[FrontLeft] = vector<double>{-1, 0, 1};
        allPossibleNeighbourParametricCoords[FrontBottomRight] = vector<double>{1, -1, 1};
        allPossibleNeighbourParametricCoords[FrontBottom] = vector<double>{0, -1, 1};
        allPossibleNeighbourParametricCoords[FrontBottomLeft] = vector<double>{-1, -1, 1};
        allPossibleNeighbourParametricCoords[Front] = vector<double>{0, 0, 1};
        allPossibleNeighbourParametricCoords[BackTopLeft] = vector<double>{-1, 1, -1};
        allPossibleNeighbourParametricCoords[BackTop] = vector<double>{0, 1, -1};
        allPossibleNeighbourParametricCoords[BackTopRight] = vector<double>{1, 1, -1};
        allPossibleNeighbourParametricCoords[BackRight] = vector<double>{1, 0, -1};
        allPossibleNeighbourParametricCoords[BackLeft] = vector<double>{-1, 0, -1};
        allPossibleNeighbourParametricCoords[BackBottomRight] = vector<double>{1, -1, -1};
        allPossibleNeighbourParametricCoords[BackBottom] = vector<double>{0, -1, -1};
        allPossibleNeighbourParametricCoords[BackBottomLeft] = vector<double>{-1, -1, -1};
        allPossibleNeighbourParametricCoords[Back] = vector<double>{0, 0, -1};
        return allPossibleNeighbourParametricCoords;
    }
    
    /*    Node* IsoParametricNeighbourFinder::getNeighbourAtPosition(unsigned nodeId, Position position) {
        auto node = _mesh->nodeFromID(nodeId);
        auto parametricCoords = node->coordinates.positionVectorPtr(Parametric);
        if (parametricCoords->size() == 1){
            parametricCoords->push_back(0);
            parametricCoords->push_back(0);
        }
        else if (parametricCoords->size() == 2){
            parametricCoords->push_back(0);
        }
        auto parametricCoordsCopy = vector<double>(*parametricCoords);
        switch (position) {
    
            case FrontTopLeft:
                (parametricCoordsCopy)[0] -= 1;
                (parametricCoordsCopy)[1] += 1;
                (parametricCoordsCopy)[2] += 1;
                break;
            case FrontTop:
                (parametricCoordsCopy)[1] += 1;
                (parametricCoordsCopy)[2] += 1;
                break;
            case FrontTopRight:
                (parametricCoordsCopy)[0] += 1;
                (parametricCoordsCopy)[1] += 1;
                (parametricCoordsCopy)[2] += 1;
                break;
            case FrontRight:
                (parametricCoordsCopy)[0] += 1;
                (parametricCoordsCopy)[2] += 1;
                break;
            case FrontLeft:
                (parametricCoordsCopy)[0] -= 1;
                (parametricCoordsCopy)[2] += 1;
                break;
            case FrontBottomRight:
                (parametricCoordsCopy)[0] += 1;
                (parametricCoordsCopy)[1] -= 1;
                (parametricCoordsCopy)[2] += 1;
                break;
            case FrontBottom:
                (parametricCoordsCopy)[1] -= 1;
                (parametricCoordsCopy)[2] += 1;
                break;
            case FrontBottomLeft:
                (parametricCoordsCopy)[0] -= 1;
                (parametricCoordsCopy)[1] -= 1;
                (parametricCoordsCopy)[2] += 1;
                break;
            case Front:
                (parametricCoordsCopy)[2] += 1;
                break;
            case BackTopLeft:
                (parametricCoordsCopy)[0] -= 1;
                (parametricCoordsCopy)[1] += 1;
                (parametricCoordsCopy)[2] -= 1;
                break;
            case BackTop:
                (parametricCoordsCopy)[1] += 1;
                (parametricCoordsCopy)[2] -= 1;
                break;
            case BackTopRight:
                (parametricCoordsCopy)[0] += 1;
                (parametricCoordsCopy)[1] += 1;
                (parametricCoordsCopy)[2] -= 1;
                break;
            case BackRight:
                (parametricCoordsCopy)[0] += 1;
                (parametricCoordsCopy)[2] -= 1;
                break;
            case BackLeft:
                (parametricCoordsCopy)[0] -= 1;
                (parametricCoordsCopy)[2] -= 1;
                break;\
            case BackBottomRight:
                (parametricCoordsCopy)[0] += 1;
                (parametricCoordsCopy)[1] -= 1;
                (parametricCoordsCopy)[2] -= 1;
                break;
            case BackBottom:
                (parametricCoordsCopy)[1] -= 1;
                (parametricCoordsCopy)[2] -= 1;
                break;
            case BackBottomLeft:
                (parametricCoordsCopy)[0] -= 1;
                (parametricCoordsCopy)[1] -= 1;
                (parametricCoordsCopy)[2] -= 1;
                break;
    
                
            default:
                break;
        }
        auto neighbourNode = _parametricCoordinatesToNodeMap->at(parametricCoordsCopy);
        delete parametricCoordsCopy;
        return neighbourNode;
    }
    
    map<vector<double>*, Node*>* IsoParametricNeighbourFinder::_createParametricCoordinatesToNodeMap() {
        for (auto &node : *_mesh->totalNodesVector) {
            auto parametricCoordinates = node->coordinates.positionVectorPtr(Parametric);
            _parametricCoordinatesToNodeMap->insert(pair<vector<double>*, Node*> (parametricCoordinates, node));
        }
        return _parametricCoordinatesToNodeMap;
    }
    
    
    
    map<Position, map<int, Node*>> IsoParametricNeighbourFinder::getNeighbourNodes(unsigned nodeId, unsigned  depth) {
        map<Position, map<int, Node*>> neighbourNodes;
        for (auto &position : positions) {
            neighbourNodes.insert(pair<Position, map<int, Node*>>(position, getCustomDepthNeighbourNodes(nodeId, position, depth)));
        }
        auto parametricCoords =
                _mesh->nodeFromID(nodeId)->coordinates.positionVectorPtr(Parametric);
        for (auto depthCounter = 0; depthCounter < depth; depthCounter++) {
            for (auto &position : positions) {
                if (position == TopLeft && 
    
            }
        }
    }*/
    
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
    

} // Discretization