//
// Created by hal9000 on 3/11/23.
//

#include "Mesh2D.h"

#include <utility>

namespace Discretization {
    
    Mesh2D::Mesh2D(shared_ptr<Array<Node*>>nodes) : Mesh(std::move(nodes)) {
        logs.startSingleObservationTimer("Mesh Initialization");
        initialize();
        logs.stopSingleObservationTimer("Mesh Initialization");
    }

    Mesh2D::~Mesh2D() {
        _cleanMeshDataStructures();
    }

    unsigned Mesh2D::dimensions() {
        return 2;
    }
    
    unsigned Mesh2D::numberOfInternalNodes() {
        return ((nodesPerDirection[One] - 2) * (nodesPerDirection[Two] - 2));
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
                (*(_nodesMatrix))(i, j)->printNode();
            }

        }
    }
    
    NumericalVector<double> Mesh2D::getNormalUnitVectorOfBoundaryNode(Position boundaryPosition, Node *node) {
        throw runtime_error("Not Implemented Yet!");
    }

    void Mesh2D::_addDBoundaryNodesToMap() {
        boundaryNodes = make_shared<map<Position, shared_ptr<vector<Node*>>>>();
                  
        auto leftBoundaryNodes = make_shared<vector<Node*>>(nodesPerDirection[Two]);
        auto rightBoundaryNodes = make_shared<vector<Node*>>(nodesPerDirection[Two]);
        for (int i = 0 ; i < nodesPerDirection[Two] ; i++) {
            (*leftBoundaryNodes)[i] = (*_nodesMatrix)(0, i);
            (*rightBoundaryNodes)[i] = (*_nodesMatrix)(nodesPerDirection[One] - 1, i);
        }
        boundaryNodes->insert( pair<Position, shared_ptr<vector<Node*>>>(Position::Left, std::move(leftBoundaryNodes)));
        boundaryNodes->insert( pair<Position, shared_ptr<vector<Node*>>>(Position::Right, std::move(rightBoundaryNodes)));
        
        auto bottomBoundaryNodes = make_shared<vector<Node*>>(nodesPerDirection[One]);
        auto topBoundaryNodes = make_shared<vector<Node*>>(nodesPerDirection[One]);
        for (int i = 0 ; i < nodesPerDirection[One] ; i++) {
            (*bottomBoundaryNodes)[i] = (*_nodesMatrix)(i, 0);
            (*topBoundaryNodes)[i] = (*_nodesMatrix)(i, nodesPerDirection[Two] - 1);
        }
        boundaryNodes->insert( pair<Position, shared_ptr<vector<Node*>>>(Position::Bottom, std::move(bottomBoundaryNodes)));
        boundaryNodes->insert( pair<Position, shared_ptr<vector<Node*>>>(Position::Top, std::move(topBoundaryNodes)));
    }
    
    void Mesh2D::_addTotalNodesToVector() {
        totalNodesVector = make_shared<vector<Node*>>(_nodesMatrix->size());
        for (int j = 0; j < nodesPerDirection[Two] ; j++){
            for (int i = 0; i < nodesPerDirection[One] ; ++i) {
                (*totalNodesVector)[i + j * nodesPerDirection[One]] = (*_nodesMatrix)(i, j);
            }
        }
    }
    
    unique_ptr<vector<Node*>> Mesh2D::getInternalNodesVector() {
        auto internalNodes = make_unique<vector<Node*>>(numberOfInternalNodes());
        auto counter = 0;
        for (int j = 1; j < nodesPerDirection[Two] - 1; j++)
            for (int i = 1; i < nodesPerDirection[One] - 1; ++i){
                internalNodes->at(counter) = (*_nodesMatrix)(i, j);
                counter++;
            }
        return internalNodes;
    }
    
    void Mesh2D::createElements(ElementType elementType, unsigned int nodesPerEdge) {
        auto numberOfElements = (nodesPerDirection[One] - 1) * (nodesPerDirection[Two] - 1);
        auto elementsVector = make_unique<vector<Element *>>(numberOfElements);
        auto counter = 0;

        auto quadrilateralNodes = [this](unsigned int i, unsigned int j) -> vector<Node*> {
            vector<Node*> nodes(4);
            nodes[0] = node(i, j);
            nodes[1] = node(i + 1, j);
            nodes[2] = node(i, j + 1);
            nodes[3] = node(i + 1, j + 1);
            return nodes;
        };
        auto triangleNodes = [this](unsigned int i, unsigned int j) -> vector<Node*> {
            if (i % 2 == 0) {
                vector<Node*> nodes(3);
                nodes[0] = node(i, j);
                nodes[1] = node(i + 1, j + 1);
                nodes[2] = node(i - 1, j + 1);
                return nodes;
            } else {
                vector<Node*> nodes(3);
                nodes[0] = node(i, j);
                nodes[1] = node(i + 1, j);
                nodes[2] = node(i, j + 1);
                return nodes;
            }
        };

        switch (elementType) {
            case Quadrilateral:
                for (int j = 0; j < nodesPerDirection[Two] - 1; j++) {
                    for (int i = 0; i < nodesPerDirection[One] - 1; ++i) {
                        vector<Node *> nodes = quadrilateralNodes(i, j);
                        auto element = new Element(counter, nodes, elementType);
                        elementsVector->at(counter) = element;
                        counter++;
                    }
                }
                break;
            case Triangle:
                for (int j = 0; j < nodesPerDirection[Two] - 1; j++) {
                    for (int i = 0; i < nodesPerDirection[One] - 1; ++i) {
                        vector<Node *> nodes = triangleNodes(i, j);
                        auto element = new Element(counter, nodes, elementType);
                        elementsVector->at(counter) = element;
                        counter++;
                    }
                }
                break;
            default:
                throw runtime_error("2D geometry only supports quadrilateral and triangle elements.");
        }

        elements = make_unique<MeshElements>(std::move(elementsVector), elementType);
    }

    void Mesh2D::storeMeshInVTKFile(const string &filePath, const string &fileName, CoordinateType coordinateType,
                                    bool StoreOnlyNodes) const {
        throw runtime_error("Not implemented yet.");
    }


}
        

