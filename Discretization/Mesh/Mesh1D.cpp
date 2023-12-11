//
// Created by hal9000 on 3/11/23.
//

#include "Mesh1D.h"

#include <utility>

namespace Discretization {
    
    Mesh1D::Mesh1D(shared_ptr<Array<Node*>>nodes) : Mesh(std::move(nodes)){
        logs.startSingleObservationTimer("Mesh Initialization");
        initialize();
        logs.stopSingleObservationTimer("Mesh Initialization");
    }
    
    Mesh1D::~Mesh1D() {
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

    NumericalVector<double> Mesh1D::getNormalUnitVectorOfBoundaryNode(Position boundaryPosition, Node *node) {
        throw runtime_error("Not implemented yet!");
    }
    
    void Mesh1D::_addDBoundaryNodesToMap() {
        boundaryNodes = make_shared<map<Position, shared_ptr<vector<Node*>>>>();
        auto leftBoundary = make_shared<vector<Node*>>(1);
        auto rightBoundary = make_shared<vector<Node*>>(1);
        (*leftBoundary)[0] = Mesh::node(0);
        (*rightBoundary)[0] = Mesh::node(Mesh::nodesPerDirection[Direction::One] - 1);
        boundaryNodes->insert(make_pair(Left, make_shared<vector<Node*>>(*leftBoundary)));
        boundaryNodes->insert(make_pair(Right, make_shared<vector<Node*>>(*rightBoundary)));
    }
    
    void Mesh1D::_addTotalNodesToVector() {
        totalNodesVector = make_shared<vector<Node*>>(numberOfTotalNodes());
        for (int i = 0; i < nodesPerDirection[Direction::One]; i++)
            (*totalNodesVector)[i] = node(i);
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