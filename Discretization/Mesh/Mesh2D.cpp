//
// Created by hal9000 on 3/11/23.
//

#include "Mesh2D.h"

#include <utility>

namespace Discretization {
    
    Mesh2D::Mesh2D(shared_ptr<Array<Node*>>nodes) : Mesh() {
        this->_nodesMatrix = std::move(nodes);
        initialize();
        _nodesMap = _createNodesMap();
    }

    Mesh2D::~Mesh2D() {
        //Deallocate all node pointers of the mesh
        for (int i = 0; i < nodesPerDirection[One] ; ++i)
            for (int j = 0; j < nodesPerDirection[Two] ; ++j){
            delete (*_nodesMatrix)(i, j);
            (*_nodesMatrix)(i, j) = nullptr;
        }
        _nodesMatrix = nullptr;
        _cleanMeshDataStructures();
    }

    unsigned Mesh2D::dimensions() {
        return 2;
    }

    SpaceEntityType Mesh2D::space() {
        return Plane;
    }

    vector<Direction> Mesh2D::directions() {
        return {One, Two};
    }

    Node *Mesh2D::node(unsigned i) {
        if (isInitialized)
            return (*_nodesMatrix)(i, 0);
        else
            throw runtime_error("Node Not Found!");
    }

    Node *Mesh2D::node(unsigned i, unsigned j) {
        if (isInitialized)
            return (*_nodesMatrix)(i, j);
        else
            throw runtime_error("Node Not Found!");
    }

    Node *Mesh2D::node(unsigned i, unsigned j, unsigned k) {
        if (k != 0 && isInitialized)
            return (*_nodesMatrix)(i, j);
        else
            throw runtime_error("A 2D Mesh can be considered a 3D mesh with 1 Node at Direction 3."
                                " Third entry must be 0.");
    }

    void Mesh2D::printMesh() {
        cout << "Number of Nodes : " << numberOfTotalNodes() << endl;
        for (int j = 0; j < nodesPerDirection[Two] ; ++j) {
            for (int i = 0; i < nodesPerDirection[One] ; ++i) {
                cout << "(i, j) : (" << i << ", " << j << ")" << endl;
                cout << "ID : (" << (*(*_nodesMatrix)(i, j)->id.global) << endl;
            }

        }
    }

    shared_ptr<map<Position, shared_ptr<vector<Node*>>>> Mesh2D::_addDBoundaryNodesToMap() {
        auto boundaryNodes = make_shared<map<Position, shared_ptr<vector<Node*>>>>();
                  
        auto *leftBoundaryNodes = new vector<Node*>(nodesPerDirection[Two]);
        auto *rightBoundaryNodes = new vector<Node*>(nodesPerDirection[Two]);
        for (int i = 0 ; i < nodesPerDirection[Two] ; i++) {
            (*leftBoundaryNodes)[i] = (*_nodesMatrix)(0, i);
            (*rightBoundaryNodes)[i] = (*_nodesMatrix)(nodesPerDirection[One] - 1, i);
        }
        boundaryNodes->insert( pair<Position, shared_ptr<vector<Node*>>>(Position::Left, leftBoundaryNodes));
        boundaryNodes->insert( pair<Position, shared_ptr<vector<Node*>>>(Position::Right, rightBoundaryNodes));
        
        auto *bottomBoundaryNodes = new vector<Node*>(nodesPerDirection[One]);
        auto *topBoundaryNodes = new vector<Node*>(nodesPerDirection[One]);
        for (int i = 0 ; i < nodesPerDirection[One] ; i++) {
            (*bottomBoundaryNodes)[i] = (*_nodesMatrix)(i, 0);
            (*topBoundaryNodes)[i] = (*_nodesMatrix)(i, nodesPerDirection[Two] - 1);
        }
        boundaryNodes->insert( pair<Position, shared_ptr<vector<Node*>>>(Position::Bottom, bottomBoundaryNodes));
        boundaryNodes->insert( pair<Position, shared_ptr<vector<Node*>>>(Position::Top, topBoundaryNodes));
        
        return boundaryNodes;
    }

    unique_ptr<vector<Node*>> Mesh2D::getInternalNodesVector() {
        auto internalNodes = make_unique<vector<Node*>>(numberOfTotalNodes());
        for (int j = 1; j < nodesPerDirection[Two] - 1; j++)
            for (int i = 1; i < nodesPerDirection[One] - 1; ++i)
                internalNodes->at(i) = (*_nodesMatrix)(i, j);
        
        return internalNodes;
    }

    shared_ptr<vector<Node*>> Mesh2D::_addTotalNodesToVector() {
        auto totalNodes = make_shared<vector<Node*>>(_nodesMatrix->size());
        for (int j = 0; j < nodesPerDirection[Two] ; j++){
            for (int i = 0; i < nodesPerDirection[One] ; ++i) {
                (*totalNodes)[i + j * nodesPerDirection[One]] = (*_nodesMatrix)(i, j);
            }
        }
        return totalNodes;
        //print node id
    }
    
    GhostPseudoMesh* Mesh2D::_createGhostPseudoMesh(unsigned ghostLayerDepth) {
        //
        auto ghostNodesPerDirection = _createNumberOfGhostNodesPerDirectionMap(ghostLayerDepth);

        auto ghostNodesList = make_shared<list<Node*>>();
        
        // Parametric coordinate 1 of nodes in the new ghost mesh
        auto nodeArrayPositionI = 0;
        // Parametric coordinate 2 of nodes in the new ghost mesh
        auto nodeArrayPositionJ = 0;
        auto nn1 = nodesPerDirection[One];
        auto nn1Ghost = ghostNodesPerDirection->at(One);
        auto nn2 = nodesPerDirection[Two];
        auto nn2Ghost = ghostNodesPerDirection->at(Two);
        
        //Create parametric coordinates to node map
        auto parametricCoordToNodeMap =  createParametricCoordToNodesMap();
        for (int j = -static_cast<int>(nn2Ghost); j < static_cast<int>(nn2) + static_cast<int>(nn2Ghost); j++) {
            for (int i = -static_cast<int>(nn1Ghost); i < static_cast<int>(nn1) + static_cast<int>(nn1Ghost); i++) {
                auto parametricCoords = vector<double>{static_cast<double>(i), static_cast<double>(j), 0};

                // If node is inside the original mesh add it to the ghost mesh Array
                if (parametricCoordToNodeMap->find(parametricCoords) == parametricCoordToNodeMap->end()) {
                    auto node = new Node();
                    node->coordinates.setPositionVector(make_shared<vector<double>>(parametricCoords), Parametric);
                    vector<double> templateCoord = {static_cast<double>(i) * specs->templateStepOne,
                                                    static_cast<double>(j) * specs->templateStepTwo};
                    // Rotate 
                    Transformations::rotate(templateCoord, specs->templateRotAngleOne);
                    // Shear
                    Transformations::shear(templateCoord, specs->templateShearOne,specs->templateShearTwo);

                    node->coordinates.setPositionVector(make_shared<vector<double>>(templateCoord), Template);
                    ghostNodesList->push_back(node);
                    parametricCoordToNodeMap->insert(pair<vector<double>, Node*>(parametricCoords, node));
                }
                nodeArrayPositionI++;
                if (nodeArrayPositionI == nn1 + 2 * nn1Ghost) {
                    nodeArrayPositionI = 0;
                    nodeArrayPositionJ++;
                }
            }
        }
        return new GhostPseudoMesh(ghostNodesList, ghostNodesPerDirection, parametricCoordToNodeMap);
    }

    shared_ptr<map<vector<double>, Node*>> Mesh2D::createParametricCoordToNodesMap() {
        auto parametricCoordToNodeMap = make_shared<map<vector<double>, Node*>>();
        for (auto& node : *totalNodesVector) {
            auto parametricCoords = node->coordinates.positionVector(Parametric);
            parametricCoords.push_back(0.0);
            parametricCoordToNodeMap->insert(pair<vector<double>, Node*>(parametricCoords, node));
        }
        return parametricCoordToNodeMap;
    }
}
        

