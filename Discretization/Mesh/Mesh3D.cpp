//
// Created by hal9000 on 3/11/23.
//

#include "Mesh3D.h"

#include <utility>
#include <fstream>

namespace Discretization {
    
    Mesh3D::Mesh3D(shared_ptr<Array<Node*>> nodes) : Mesh(std::move(nodes)) {
        logs.startSingleObservationTimer("Mesh Initialization");
        initialize();
        logs.stopSingleObservationTimer("Mesh Initialization");
    }
    
    Mesh3D::~Mesh3D() {
        _cleanMeshDataStructures();
    }
    
    unsigned Mesh3D::dimensions() {
        return 3;
    }
    
    SpaceEntityType Mesh3D::space() {
        return Volume;
    }

    vector<Direction> Mesh3D::directions() {
        return {One, Two, Three};
    }
    
    unsigned Mesh3D::numberOfInternalNodes() {
        return ((nodesPerDirection[One] - 2) * (nodesPerDirection[Two] - 2) * (nodesPerDirection[Three] - 2));
    }
    
    Node* Mesh3D::node(unsigned i) {
        if (isInitialized)
            return (*_nodesMatrix)(i);
        else
            throw runtime_error("Node Not Found!");
    }
    
    Node* Mesh3D::node(unsigned i, unsigned j) {
        if (isInitialized)
            return (*_nodesMatrix)(i, j);
        else
            throw runtime_error("Node Not Found!");
    }
    
    Node* Mesh3D::node(unsigned i, unsigned j, unsigned k) {
        if (isInitialized)
            return (*_nodesMatrix)(i, j, k);
        else
            throw runtime_error("Node Not Found!");
    }
    
    void Mesh3D::printMesh() {
        for (int k = 0 ; k < nodesPerDirection[Three] ; k++)
            for (int j = 0 ; j < nodesPerDirection[Two] ; j++)
                for (int i = 0 ; i < nodesPerDirection[One] ; i++) {
                    (*_nodesMatrix)(i, j, k)->printNode();
                }   
    }
    
    NumericalVector<double> getNormalUnitVectorOfBoundaryNode(Position boundaryPosition, Node *node) {
        return {};
    }

    void Mesh3D::_addDBoundaryNodesToMap() {
        
        boundaryNodes = make_shared<map<Position, shared_ptr<vector<Node*>>>>();

        auto bottomBoundaryNodes = make_shared<vector<Node*>>(nodesPerDirection[One] * nodesPerDirection[Two]);
        auto topBoundaryNodes = make_shared<vector<Node*>>(nodesPerDirection[One] * nodesPerDirection[Two]);
        auto index = 0;
        for (int i = 0 ; i < nodesPerDirection[Two] ; i++) {
            for (int j = 0 ; j < nodesPerDirection[One] ; j++) {
                (*bottomBoundaryNodes)[index] = (*_nodesMatrix)(j, i, 0);
                (*topBoundaryNodes)[index] = (*_nodesMatrix)(j, i, nodesPerDirection[Three] - 1);
                index++;
            }
        }
        boundaryNodes->insert(make_pair(Position::Bottom, (bottomBoundaryNodes)));
        boundaryNodes->insert(make_pair(Position::Top, (topBoundaryNodes)));
        index = 0;
        auto leftBoundaryNodes = make_shared<vector<Node*>>(nodesPerDirection[Two] * nodesPerDirection[Three]);
        auto rightBoundaryNodes = make_shared<vector<Node*>>(nodesPerDirection[Two] * nodesPerDirection[Three]);
        for (int i = 0 ; i < nodesPerDirection[Three] ; i++) {
            for (int j = 0 ; j < nodesPerDirection[Two] ; j++) {
                (*leftBoundaryNodes)[index] = (*_nodesMatrix)(0, j, i);
                (*rightBoundaryNodes)[index] = (*_nodesMatrix)(nodesPerDirection[One] - 1, j, i);
                index++;
            }
        }
        boundaryNodes->insert( pair<Position, shared_ptr<vector<Node*>>>(Position::Left, (leftBoundaryNodes)));
        boundaryNodes->insert( pair<Position, shared_ptr<vector<Node*>>>(Position::Right, (rightBoundaryNodes)));

        
        index = 0;
        auto frontBoundaryNodes = make_shared<vector<Node*>>(nodesPerDirection[Three] * nodesPerDirection[One]);
        auto backBoundaryNodes = make_shared<vector<Node*>>(nodesPerDirection[Three] * nodesPerDirection[One]);
        for (int i = 0 ; i < nodesPerDirection[Three] ; i++) {
            for (int j = 0 ; j < nodesPerDirection[One] ; j++) {
                (*frontBoundaryNodes)[index] = (*_nodesMatrix)(j, nodesPerDirection[Two] - 1, i);
                (*backBoundaryNodes)[index] = (*_nodesMatrix)(j, 0, i);
                index++;
            }
        }
        
        boundaryNodes->insert( pair<Position, shared_ptr<vector<Node*>>>(Position::Front, (frontBoundaryNodes)));
        boundaryNodes->insert( pair<Position, shared_ptr<vector<Node*>>>(Position::Back, (backBoundaryNodes)));
    }

    unique_ptr<vector<Node*>> Mesh3D::getInternalNodesVector() {
        auto internalNodes = make_unique<vector<Node*>>(numberOfInternalNodes());
        auto counter = 0;
        for (int k = 1; k < nodesPerDirection[Three] - 1; k++){
            for (int j = 1; j < nodesPerDirection[Two] - 1; j++){
                for (int i = 1; i < nodesPerDirection[One] - 1; i++) {
                    internalNodes->at(counter) = _nodesMatrix->at(i, j, k);
                    counter++;
                }
            }
        }
        return std::move(internalNodes);
    }

    void Mesh3D::_addTotalNodesToVector() {
        totalNodesVector = make_shared<vector<Node*>>(numberOfTotalNodes());
        for (int k = 0; k < nodesPerDirection[Three]; k++){
            for (int j = 0; j < nodesPerDirection[Two]; j++){
                for (int i = 0; i < nodesPerDirection[One]; ++i) {
                    totalNodesVector->at(i + j * nodesPerDirection[One] + k * nodesPerDirection[One] * nodesPerDirection[Two]) = (*_nodesMatrix)(i, j, k);    
                }
            }
        }
    }
    
    //TODO it works only with template mesh
    NumericalVector<double> Mesh3D::getNormalUnitVectorOfBoundaryNode(Position boundaryPosition, Node *node) {
        map<Position, vector<Direction>> directionsOfBoundaries = {
                {Top, {One, Two}},
                {Bottom, {Two, One}},
                {Right, {Two, Three}},
                {Left, {Three, Two}},
                {Back, {Three, One}},
                {Front, {One, Three}}
        };
        //Check if boundaryPosition exists in map
        if (directionsOfBoundaries.find(boundaryPosition) != directionsOfBoundaries.end()){
            Direction direction1 = directionsOfBoundaries[boundaryPosition][0];
            Direction direction2 = directionsOfBoundaries[boundaryPosition][1];

            NumericalVector<double> covariantBaseVector1 = metrics->at(node)->covariantBaseVectors->at(direction1);
            NumericalVector<double> covariantBaseVector2 = metrics->at(node)->covariantBaseVectors->at(direction2);
            NumericalVector<double> normalUnitVector = NumericalVector<double>(3);
            covariantBaseVector1.crossProduct(covariantBaseVector2, normalUnitVector);
            normalUnitVector.normalize();
            
           /* cout<<*node->id.global<<endl;
            cout<<boundaryPosition<<" "<<normalUnitVector[0]<<" "<<normalUnitVector[1]<<" "<<normalUnitVector[2]<<endl;*/
            
            return normalUnitVector;
        }    
        else {
            throw invalid_argument("Boundary position not found");
        }
    }

    void Mesh3D::createElements(ElementType elementType, unsigned int nodesPerEdge) {
        auto numberOfElements = (nodesPerDirection[One] - 1) * (nodesPerDirection[Two] - 1) * (nodesPerDirection[Three] - 1);
        auto elementsVector = make_unique<vector<Element *>>(numberOfElements);
        auto counter = 0;

        auto hexahedron = [this](unsigned int i, unsigned int j, unsigned k) -> vector<Node*> {
            vector<Node*> nodes(8);
            nodes[0] = node(i, j, k);
            nodes[1] = node(i + 1, j, k);
            nodes[2] = node(i, j + 1, k);
            nodes[3] = node(i + 1, j + 1, k);
            nodes[4] = node(i, j, k + 1);
            nodes[5] = node(i + 1, j, k + 1);
            nodes[6] = node(i, j + 1, k + 1);
            nodes[7] = node(i + 1, j + 1, k + 1);
            return nodes;
        };
        auto wedge = [this](unsigned int i, unsigned int j, unsigned k) -> vector<Node*> {
            if (i % 2 == 0) {
                vector<Node*> nodes(6);
                nodes[0] = node(i, j, k);
                nodes[1] = node(i + 1, j, k);
                nodes[2] = node(i, j + 1, k);
                nodes[3] = node(i + 1, j + 1, k);
                nodes[4] = node(i, j + 1, k + 1);
                nodes[5] = node(i + 1, j + 1, k + 1);
                return nodes;

            } else {
                vector<Node*> nodes(6);
                nodes[0] = node(i, j, k);
                nodes[1] = node(i, j + 1, k);
                nodes[2] = node(i, j, k + 1);
                nodes[3] = node(i + 1, j, k + 1);
                nodes[4] = node(i, j + 1, k + 1);
                nodes[5] = node(i + 1, j + 1, k + 1);
                return nodes;
            }
        };

        switch (elementType) {
            case Hexahedron:
                for (int k = 0; k < nodesPerDirection[Three] - 1; ++k) {
                    for (int j = 0; j < nodesPerDirection[Two] - 1; ++j) {
                        for (int i = 0; i < nodesPerDirection[One] - 1; ++i) {
                            vector<Node *> nodes = hexahedron(i, j, k);
                            auto element = new Element(counter, nodes, elementType);
                            elementsVector->at(counter) = element;
                            counter++;
                        }
                    }
                }
                break;
            case Wedge:
                for (int k = 0; k < nodesPerDirection[Three] - 1; ++k) {
                    for (int j = 0; j < nodesPerDirection[Two] - 1; ++j) {
                        for (int i = 0; i < nodesPerDirection[One] - 1; ++i) {
                            vector<Node *> nodes = wedge(i, j, k);
                            auto element = new Element(counter, nodes, elementType);
                            elementsVector->at(counter) = element;
                            counter++;
                        }
                    }
                }
                break;
            default:
                throw runtime_error("2D geometry only supports quadrilateral and triangle elements.");
        }

        elements = make_unique<MeshElements>(std::move(elementsVector), elementType);
    }


    void Mesh3D::storeMeshInVTKFile(const string &filePath, const string &fileName, CoordinateType coordinateType,
                                    bool StoreOnlyNodes) const {
        ofstream outputFile(filePath + fileName);

        // Header
        outputFile << "# vtk DataFile Version 3.0 \n";
        outputFile << "vtk output \n";
        outputFile << "ASCII \n";
        outputFile << "DATASET STRUCTURED_GRID \n";

        // Assuming the mesh is nx x ny x nz, specify the dimensions
        unsigned int nx = nodesPerDirection.at(One);
        unsigned int ny = nodesPerDirection.at(Two);
        unsigned int nz = nodesPerDirection.at(Three);

        outputFile << "DIMENSIONS " << nx << " " << ny<< " " << nz<< "\n";

        // Points
        outputFile << "POINTS " << totalNodesVector->size() << " double\n";
        for (auto &node: *totalNodesVector) {
            auto coordinates = node->coordinates.getPositionVector3D(coordinateType);
            outputFile << coordinates[0] << " " << coordinates[1] << " " << coordinates[2] << "\n";
        }

        outputFile.close();
    }



} // Discretization