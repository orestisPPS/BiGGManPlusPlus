//
// Created by hal9000 on 4/3/23.
//

#include "IsoparametricNodeHoodGraph.h"

namespace Discretization {
    
    IsoParametricNodeHoodGraph::IsoParametricNodeHoodGraph(Node* node, unsigned graphDepth, Mesh* mesh,
                                                           map<vector<double>, Node *> *nodeMap) :
            _node(node), _mesh(mesh), _graphDepth(graphDepth), _nodeMap(nodeMap){
        
        neighborhood = new map<Position,vector<Node*>>();
        auto nodeCoords = node->coordinates.positionVector(Parametric);
        if (nodeCoords.size() == 2) {
            nodeCoords.push_back(0);
        }
        else if (nodeCoords.size() == 1) {
            nodeCoords.push_back(0);
            nodeCoords.push_back(0);
        }
        for (int i = 1; i <= _graphDepth + 1; ++i) {
            _findIDepthNeighborhood(i, nodeCoords);
        }
    }
    
    void IsoParametricNodeHoodGraph::_findINeighborhoodRecursive() {
    
    }
    
    void IsoParametricNodeHoodGraph::_findIDepthNeighborhood(unsigned int depth, vector<double>& nodeCoords) {

        auto nn1 = _mesh->numberOfNodesPerDirection[One];
        auto nn2 = _mesh->numberOfNodesPerDirection[Two];
        auto nn3 = _mesh->numberOfNodesPerDirection[Three];
        
        auto id = *_node->id.global;
        auto k = id / (nn1 * nn2);
        
        auto parametricCoords = vector<double>{0, 0, 0};
        //TopLeft
        parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] + depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(TopLeft, parametricCoords, depth);
        //Top
        parametricCoords = {nodeCoords[0], nodeCoords[1] + depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Top, parametricCoords, depth);
        //TopRight
        parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] + depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(TopRight, parametricCoords, depth);
        //Left
        parametricCoords = {nodeCoords[0] - depth, nodeCoords[1], nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Left, parametricCoords, depth);
        //Right
        parametricCoords = {nodeCoords[0] + depth, nodeCoords[1], nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Right, parametricCoords, depth);
        //BottomRight
        parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] - depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(BottomRight, parametricCoords, depth);
        //Bottom
        parametricCoords = {nodeCoords[0], nodeCoords[1] - depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(Bottom, parametricCoords, depth);
        //BottomLeft
        parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] - depth, nodeCoords[2]};
        _addNeighbourNodeIfParametricCoordsExist(BottomLeft, parametricCoords, depth);

        if (k > 0){
            //FrontTopLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] + depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(FrontTopLeft, parametricCoords, depth);

            //FrontTop
            parametricCoords = {nodeCoords[0], nodeCoords[1] + depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(FrontTop, parametricCoords, depth);
            //FrontTopRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] + depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(FrontTopRight, parametricCoords, depth);
            //FrontRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1], nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(FrontRight, parametricCoords, depth);
            //Front
            parametricCoords = {nodeCoords[0], nodeCoords[1], nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(Front, parametricCoords, depth);
            //FrontLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1], nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(FrontLeft, parametricCoords, depth);
            //FrontBottomRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] - depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(FrontBottomRight, parametricCoords, depth);
            //FrontBottom
            parametricCoords = {nodeCoords[0], nodeCoords[1] - depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(FrontBottom, parametricCoords, depth);
            //FrontBottomLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] - depth, nodeCoords[2] + depth};
            _addNeighbourNodeIfParametricCoordsExist(FrontBottomLeft, parametricCoords, depth);
        }
        
        if (k < nn3 - 1) {
            //BackTopLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] + depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(BackTopLeft, parametricCoords, depth);
            //BackTop
            parametricCoords = {nodeCoords[0], nodeCoords[1] + depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(BackTop, parametricCoords, depth);
            //BackTopRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] + depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(BackTopRight, parametricCoords, depth);
            //BackLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1], nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(BackLeft, parametricCoords, depth);
            //Back
            parametricCoords = {nodeCoords[0], nodeCoords[1], nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(Back, parametricCoords, depth);
            //BackRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1], nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(BackRight, parametricCoords, depth);
            //BackBottomLeft
            parametricCoords = {nodeCoords[0] - depth, nodeCoords[1] - depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(BackBottomLeft, parametricCoords, depth);
            //BackBottom
            parametricCoords = {nodeCoords[0], nodeCoords[1] - depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(BackBottom, parametricCoords, depth);
            //BackBottomRight
            parametricCoords = {nodeCoords[0] + depth, nodeCoords[1] - depth, nodeCoords[2] - depth};
            _addNeighbourNodeIfParametricCoordsExist(BackBottomRight, parametricCoords, depth);
        }
        
    }
    
    void IsoParametricNodeHoodGraph:: _addNeighbourNodeIfParametricCoordsExist(Position position,
                                       vector<double>& parametricCoords, unsigned currentDepth){
        if (_nodeMap->find(parametricCoords) != _nodeMap->end()){
            if (neighborhood->find(position) == neighborhood->end()){
                neighborhood->insert(pair<Position, vector<Node*>>(position, vector<Node*>()));
            }
            if (neighborhood->at(position).empty()){
                neighborhood->at(position) = vector<Node*>();
            }
            neighborhood->at(position).push_back(_nodeMap->at(parametricCoords));
        }
    }


} // Node