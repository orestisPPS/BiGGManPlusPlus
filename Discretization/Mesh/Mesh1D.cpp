//
// Created by hal9000 on 3/11/23.
//

#include "Mesh1D.h"

namespace Discretization {
    
    Mesh1D::Mesh1D(Array<Node *> *nodes) : Mesh(){
        this->_nodesMatrix = nodes;
        initialize();
        _nodesMap = createNodesMap();
    }
    
    Mesh1D::~Mesh1D() {
        //Deallocate all node pointers of the mesh
        for (int i = 0; i < numberOfNodesPerDirection[One] ; ++i) {
            delete (*_nodesMatrix)(i);
            (*_nodesMatrix)(i) = nullptr;
        }
        delete _nodesMatrix;
        _nodesMatrix = nullptr;
        
        cleanMeshDataStructures();
        
    }
    
    unsigned Mesh1D::dimensions() {
        return 1;
    }
    
    SpaceEntityType Mesh1D::space() {
        return Axis;
    }

    Node* Mesh1D::node(unsigned i) {
        if (_nodesMatrix != nullptr)
            return (*_nodesMatrix)(i);
        else
            throw runtime_error("Node Not Found!");
    }

    Node* Mesh1D::node(unsigned i, unsigned j) {
        if (j != 0 && isInitialized)
            return (*_nodesMatrix)(i);
        else 
            throw runtime_error("A 1D Mesh can be considered a 2D mesh with 1 Node at Direction 2."
                                " Second entry must be 0.");
    }
    
    Node* Mesh1D::node(unsigned i, unsigned j, unsigned k) {
        if (j != 1 && k != 0 && isInitialized)
            return (*_nodesMatrix)(i);
        else
            throw runtime_error("A 1D Mesh can be considered a 3D mesh with 1 Node at Directions 2 and 3."
                                " Second and third entries must be 0.");
    }
    
    void Mesh1D::printMesh() {
        cout << "Mesh1D" << endl;
        for (int i = 0 ; i < numberOfNodesPerDirection[Direction::One] ; i++) {
            (*_nodesMatrix)(i)->printNode();
        }
    }
    

    
    map<Position, vector<Node*>*> *Mesh1D::addDBoundaryNodesToMap() {
        auto boundaries = new map<Position, vector<Node*>*>();
        auto leftBoundary = new vector<Node*>(1);
        auto rightBoundary = new vector<Node*>(1);
        leftBoundary->push_back(Mesh::node(0));
        rightBoundary->push_back(Mesh::node(Mesh::numberOfNodesPerDirection[Direction::One] - 1));
        boundaries->insert( pair<Position, vector<Node*>*>(Position::Left, leftBoundary));
        return boundaries;
    }
    
    vector<Node*>* Mesh1D::addInternalNodesToVector() {
        auto internalNodes = new vector<Node*>();
        for (int i = 1; i < numberOfNodesPerDirection[Direction::One] - 1; i++) {
            internalNodes->push_back(Mesh::node(i));
        }
        return internalNodes;
    }
    
    vector<Node*>* Mesh1D::addTotalNodesToVector() {
        auto totalNodes = new vector<Node*>(_nodesMatrix->size());
        for (int i = 0; i < numberOfNodesPerDirection[Direction::One]; i++) {
            totalNodes->push_back(Mesh::node(i));
        }
        return totalNodes;
    }
    
    GhostPseudoMesh* Mesh1D::createGhostPseudoMesh(unsigned ghostLayerDepth) {
        //
        auto ghostNodesPerDirection = createNumberOfGhostNodesPerDirectionMap(ghostLayerDepth);

        auto ghostNodesList = new list<Node*>();

        // Parametric coordinate 1 of nodes in the new ghost mesh
        auto nodeArrayPositionI = 0;
        
        auto nn1 = numberOfNodesPerDirection[One];
        auto nn1Ghost = ghostNodesPerDirection->at(One);

        //Create parametric coordinates to node map
        auto parametricCoordToNodeMap =  createParametricCoordToNodesMap();
        for (int i = -static_cast<int>(nn1Ghost); i < static_cast<int>(nn1) + static_cast<int>(nn1Ghost); i++) {
                auto parametricCoords = vector<double>{static_cast<double>(i), 0, 0};
                // If node is inside the original mesh add it to the ghost mesh Array
                if (parametricCoordToNodeMap->find(parametricCoords) == parametricCoordToNodeMap->end()) {
                    auto node = new Node();
                    node->coordinates.setPositionVector(parametricCoords, Parametric);
                    vector<double> templateCoord = {static_cast<double>(i) * specs->templateStepOne};
                    node->coordinates.setPositionVector(templateCoord, Template);
                    ghostNodesList->push_back(node);
                }
                nodeArrayPositionI++;
        }
        return new GhostPseudoMesh(ghostNodesList, ghostNodesPerDirection, parametricCoordToNodeMap);
    }
    
    map<vector<double>, Node*>* Mesh1D::createParametricCoordToNodesMap() {
        auto parametricCoordToNodeMap = new map<vector<double>, Node*>();
        for (auto& node : *totalNodesVector) {
            auto parametricCoords = node->coordinates.positionVector(Parametric);
            parametricCoords.push_back(0.0);
            parametricCoords.push_back(0.0);
            parametricCoordToNodeMap->insert(pair<vector<double>, Node*>(parametricCoords, node));
        }
        return parametricCoordToNodeMap;
    }
    
    Metrics* Mesh1D::calculateNodeMetrics(Discretization::Node *node,
                                          PositioningInSpace::CoordinateType coordinateSystem) {
        return nullptr;
    }
    
    
    
    
    
} // Discretization