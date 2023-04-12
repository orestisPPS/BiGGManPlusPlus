//
// Created by hal9000 on 4/2/23.
//

#include "IsoParametricNeighbourFinder.h"

namespace Discretization {

    IsoParametricNeighbourFinder::IsoParametricNeighbourFinder(Mesh *mesh){
        this->_mesh = mesh;
        this->_parametricCoordinatesToNodeMap = _createParametricCoordinatesToNodeMap();
    }
    
    IsoParametricNeighbourFinder::IsoParametricNeighbourFinder(Discretization::Mesh *mesh,
                                                               map<vector<double>, Discretization::Node *> *parametricCoordinatesToNodeMap) {
        this->_mesh = mesh;
        this->_parametricCoordinatesToNodeMap = parametricCoordinatesToNodeMap;
    }
    
    IsoParametricNeighbourFinder::~IsoParametricNeighbourFinder() {
        cout << "IsoParametricNeighbourFinder destructor called" << endl;
    }
    
    IsoParametricNodeGraph IsoParametricNeighbourFinder::getIsoParametricNodeGraph(Node *node, int graphDepth) {
        auto isoParametricNodeGraph = IsoParametricNodeGraph(node, graphDepth, _mesh,
                                                   _parametricCoordinatesToNodeMap);
        return isoParametricNodeGraph;
    }


    map<vector<double>, Node *>* IsoParametricNeighbourFinder::_createParametricCoordinatesToNodeMap() {
        auto parametricCoordMap = new map<vector<double>, Node *>();
        for (auto &node: *_mesh->totalNodesVector) {
            auto parametricCoordinates = node->coordinates.positionVector(Parametric);
            if (parametricCoordinates.size() == 2) {
                parametricCoordinates.push_back(0);
            } else if (parametricCoordinates.size() == 1) {
                parametricCoordinates.push_back(0);
                parametricCoordinates.push_back(0);
            }
            parametricCoordMap->insert(pair<vector<double>, Node *>(parametricCoordinates, node));
        }
        return parametricCoordMap;
    }
} // Discretization

