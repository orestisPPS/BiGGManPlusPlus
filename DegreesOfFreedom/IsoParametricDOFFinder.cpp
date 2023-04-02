//
// Created by hal9000 on 4/2/23.
//

#include "IsoParametricDOFFinder.h"

namespace DegreesOfFreedom {
    IsoParametricDOFFinder::IsoParametricDOFFinder(AnalysisDegreesOfFreedom *analysisDOF, Mesh *mesh) :
                            _analysisDOF(analysisDOF), _mesh(mesh){
        _parametricCoordinatesToNodeMap = _createParametricCoordinatesToNodeMap();
        _nodeToParametricCoordinatesMap = _createNodeToParametricCoordinatesMap();
        _dofToParametricCoordinatesMap = _createDOFToParametricCoordinatesMap();
    }

    map<vector<double>, Node*> IsoParametricDOFFinder::_createParametricCoordinatesToNodeMap() {
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

    map<Node*, vector<double>> IsoParametricDOFFinder::_createNodeToParametricCoordinatesMap() {
        auto nodeToParametricCoordMap = map<Node*, vector<double>>();
        for (auto &node : _parametricCoordinatesToNodeMap) {
            nodeToParametricCoordMap.insert(pair<Node*, vector<double>> (node.second, node.first));
        }
        return nodeToParametricCoordMap;
    }

    map<DegreeOfFreedom *, vector<double>> IsoParametricDOFFinder::_createDOFToParametricCoordinatesMap() {
        auto dofToParametricCoordMap = map<DegreeOfFreedom*, vector<double>>();
        for (auto &node : _nodeToParametricCoordinatesMap) {\
            for (auto &dof : *node.first->degreesOfFreedom){
                dofToParametricCoordMap.insert(pair<DegreeOfFreedom*, vector<double>> (dof, node.second));
            }
        }
        return dofToParametricCoordMap;
    }


} // DeggreesOfFreedom