//
// Created by hal9000 on 3/11/23.
//

#include "Mesh1D.h"

#include <utility>

namespace Discretization {
    
    Mesh1D::Mesh1D(shared_ptr<Array<Node*>>nodes) : Mesh(){
        this->_nodesMatrix = std::move(nodes);
        initialize();
        _nodesMap = _createNodesMap();
    }
    
    Mesh1D::~Mesh1D() {
        //Deallocate all node pointers of the mesh
        for (int i = 0; i < nodesPerDirection[One] ; ++i) {
            delete (*_nodesMatrix)(i);
            (*_nodesMatrix)(i) = nullptr;
        }
        _nodesMatrix = nullptr;

        _cleanMeshDataStructures();
        
    }
    
    unsigned Mesh1D::dimensions() {
        return 1;
    }
    
    unsigned Mesh1D::numberOfInternalNodes() {
        return nodesPerDirection[One] - 2;
    }
    
    SpaceEntityType Mesh1D::space() {
        return Axis;
    }
    
    vector<Direction> Mesh1D::directions() {
        return {One};
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
                                "Second entry must be 0.");
    }
    
    Node* Mesh1D::node(unsigned i, unsigned j, unsigned k) {
        if (j != 1 && k != 0 && isInitialized)
            return (*_nodesMatrix)(i);
        else
            throw runtime_error("A 1D Mesh can be considered a 3D mesh with 1 Node at Directions 2 and 3."
                                "Second and third entries must be 0.");
    }
    
    unique_ptr<vector<Node*>> Mesh1D::getInternalNodesVector() {
        auto internalNodes = make_unique<vector<Node*>>(numberOfInternalNodes());
        for (int i = 1; i < nodesPerDirection[Direction::One] - 1; i++) 
            *internalNodes->at(i) = *node(i);
        return internalNodes;
    }
    
    
    void Mesh1D::printMesh() {
        cout << "Mesh1D" << endl;
        for (int i = 0 ; i < nodesPerDirection[Direction::One] ; i++) {
            (*_nodesMatrix)(i)->printNode();
        }
    }

    vector<double> Mesh1D::getNormalUnitVectorOfBoundaryNode(Position boundaryPosition, Node *node) {
        throw runtime_error("Not implemented yet!");
    }
    
    shared_ptr<map<Position, shared_ptr<vector<Node*>>>> Mesh1D::_addDBoundaryNodesToMap() {
        auto boundaries = make_shared<map<Position, shared_ptr<vector<Node*>>>>();
        auto leftBoundary = new vector<Node*>(1);
        auto rightBoundary = new vector<Node*>(1);
        leftBoundary->push_back(Mesh::node(0));
        rightBoundary->push_back(Mesh::node(Mesh::nodesPerDirection[Direction::One] - 1));
        boundaries->insert( pair<Position, shared_ptr<vector<Node*>>>(Position::Left, leftBoundary));
        return boundaries;
    }
    
    shared_ptr<vector<Node*>> Mesh1D::_addTotalNodesToVector() {
        auto totalNodes = make_shared<vector<Node*>>(numberOfTotalNodes());
        for (int i = 0; i < nodesPerDirection[Direction::One]; i++) {
            totalNodes->at(i) = node(i);
        }
        return totalNodes;
    }
    
    GhostPseudoMesh* Mesh1D::_createGhostPseudoMesh(unsigned ghostLayerDepth) {
        //
        auto ghostNodesPerDirection = _createNumberOfGhostNodesPerDirectionMap(ghostLayerDepth);

        auto ghostNodesList = make_shared<list<Node*>>();

        // Parametric coordinate 1 of nodes in the new ghost mesh
        auto nodeArrayPositionI = 0;
        
        auto nn1 = nodesPerDirection[One];
        auto nn1Ghost = ghostNodesPerDirection->at(One);

        //Create parametric coordinates to node map
        auto parametricCoordToNodeMap =  createParametricCoordToNodesMap();
        for (int i = -static_cast<int>(nn1Ghost); i < static_cast<int>(nn1) + static_cast<int>(nn1Ghost); i++) {
                auto parametricCoords = vector<double>{static_cast<double>(i), 0, 0};
                // If node is inside the original mesh add it to the ghost mesh Array
                if (parametricCoordToNodeMap->find(parametricCoords) == parametricCoordToNodeMap->end()) {
                    auto node = new Node();
                    node->coordinates.setPositionVector(make_shared<vector<double>>(parametricCoords), Parametric);
                    vector<double> templateCoord = {static_cast<double>(i) * specs->templateStepOne};
                    node->coordinates.setPositionVector(make_shared<vector<double>>(templateCoord), Template);
                    ghostNodesList->push_back(node);
                }
                nodeArrayPositionI++;
        }
        return new GhostPseudoMesh(ghostNodesList, ghostNodesPerDirection, parametricCoordToNodeMap);
    }
    
    shared_ptr<map<vector<double>, Node*>> Mesh1D::createParametricCoordToNodesMap() {
        auto parametricCoordToNodeMap = make_shared<map<vector<double>, Node*>>();
        for (auto& node : *totalNodesVector) {
            auto parametricCoords = node->coordinates.positionVector(Parametric);
            parametricCoords.push_back(0.0);
            parametricCoords.push_back(0.0);
            parametricCoordToNodeMap->insert(pair<vector<double>, Node*>(parametricCoords, node));
        }
        return parametricCoordToNodeMap;
    }

    void Mesh1D::createElements(ElementType elementType, unsigned int nodesPerEdge) {
        if (elementType == Line){
            unsigned numberOfElements = nodesPerDirection[Direction::One] - 1;
            vector<Element*> elements = vector<Element*>(numberOfElements);
            for (int i = 0; i < numberOfElements; i++) {
                vector<Node*> nodes = {node(i), node(i + 1)};
                elements[i] = new Element(i, nodes, elementType);
            }
        }
        else
            throw invalid_argument("Mesh1D can only create Line elements!");
    }

    void Mesh1D::storeMeshInVTKFile(const string &filePath, const string &fileName, CoordinateType coordinateType,
                                    bool StoreOnlyNodes) const {
        throw runtime_error("Not implemented yet!");
    }


} // Discretization