//
// Created by hal9000 on 4/2/23.
//

#include "IsoParametricNeighbourFinder.h"

namespace Discretization {

    IsoParametricNeighbourFinder::IsoParametricNeighbourFinder(Mesh *mesh){
        this->_mesh = mesh;
        this->_parametricCoordinatesToNodeMap = _createParametricCoordinatesToNodeMap();
        maxMagnitude = Utility::Calculators::magnitude(vector<double>{
                static_cast<double>(mesh->numberOfNodesPerDirection[PositioningInSpace::One] - 1),
                static_cast<double>(mesh->numberOfNodesPerDirection[PositioningInSpace::Two] - 1),
                static_cast<double>(mesh->numberOfNodesPerDirection[PositioningInSpace::Three] - 1)});
        //auto hoodTest = getAllNeighbourNodes(6, 1);
        //auto dofHoodTest = getAllNeighbourDOF(6, 1);
        auto graph = new IsoParametricNodeHoodGraph(_mesh->nodeFromID(16), 3, mesh, _parametricCoordinatesToNodeMap);
    }

    IsoParametricNeighbourFinder::~IsoParametricNeighbourFinder() {
        cout << "IsoParametricNeighbourFinder destructor called" << endl;
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

