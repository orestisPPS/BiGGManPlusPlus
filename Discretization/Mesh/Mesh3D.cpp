//
// Created by hal9000 on 3/11/23.
//

#include "Mesh3D.h"

#include <utility>
#include <fstream>

namespace Discretization {
    
    Mesh3D::Mesh3D(shared_ptr<Array<Node*>> nodes) : Mesh(){
        this->_nodesMatrix = std::move(nodes);
        initialize();
        _nodesMap = _createNodesMap();
    }
    
    Mesh3D::~Mesh3D() {
        for (int i = 0; i < nodesPerDirection[One] ; ++i)
            for (int j = 0; j < nodesPerDirection[Two] ; ++j)
                for (int k = 0; k < nodesPerDirection[Three] ; ++k){
                delete (*_nodesMatrix)(i, j, k);
                (*_nodesMatrix)(i, j, k) = nullptr;
            }
        _nodesMatrix = nullptr;

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
    
    vector<double> getNormalUnitVectorOfBoundaryNode(Position boundaryPosition, Node *node) {

    }

    shared_ptr<map<Position, shared_ptr<vector<Node*>>>> Mesh3D::_addDBoundaryNodesToMap() {
        auto boundaryNodes = make_shared<map<Position, shared_ptr<vector<Node*>>>>();
        
        auto bottomNodes = new vector<Node*>();
        auto topNodes = new vector<Node*>();
        for (int i = 0 ; i < nodesPerDirection[Two] ; i++) {
            for (int j = 0 ; j < nodesPerDirection[One] ; j++) {
                bottomNodes->push_back((*_nodesMatrix)(j, i, 0));
                topNodes->push_back((*_nodesMatrix)(j, i, nodesPerDirection[Three] - 1));
            }
        }
        boundaryNodes->insert(pair<Position, shared_ptr<vector<Node*>>>(Bottom, bottomNodes));
        boundaryNodes->insert(pair<Position, shared_ptr<vector<Node*>>>(Top, topNodes));
        
        auto leftNodes = new vector<Node*>();
        auto rightNodes = new vector<Node*>();
        for (int i = 0 ; i < nodesPerDirection[Three] ; i++) {
            for (int j = 0 ; j < nodesPerDirection[Two] ; j++) {
                leftNodes->push_back((*_nodesMatrix)(0, j, i));
                rightNodes->push_back((*_nodesMatrix)(nodesPerDirection[One] - 1, j, i));
            }
        }
        boundaryNodes->insert(pair<Position, shared_ptr<vector<Node*>>>(Left, leftNodes));
        boundaryNodes->insert(pair<Position, shared_ptr<vector<Node*>>>(Right, rightNodes));
    
        auto frontNodes = new vector<Node*>();
        auto backNodes = new vector<Node*>();
        for (int i = 0 ; i < nodesPerDirection[Three] ; i++) {
            for (int j = 0 ; j < nodesPerDirection[One] ; j++) {
                frontNodes->push_back((*_nodesMatrix)(j, nodesPerDirection[Two] - 1, i));
                backNodes->push_back((*_nodesMatrix)(j, 0, i));
            }
        }
        boundaryNodes->insert(pair<Position, shared_ptr<vector<Node*>>>(Front, frontNodes));
        boundaryNodes->insert(pair<Position, shared_ptr<vector<Node*>>>(Back, backNodes));
        
        return boundaryNodes;
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
        return internalNodes;
    }

    shared_ptr<vector<Node*>> Mesh3D::_addTotalNodesToVector() {
        auto totalNodes = make_shared<vector<Node*>>(numberOfTotalNodes());
        for (int k = 0; k < nodesPerDirection[Three]; k++){
            for (int j = 0; j < nodesPerDirection[Two]; j++){
                for (int i = 0; i < nodesPerDirection[One]; ++i) {
                    totalNodes->at(i + j * nodesPerDirection[One] + k * nodesPerDirection[One] * nodesPerDirection[Two]) = (*_nodesMatrix)(i, j, k);    
                }
            }
        }
        return totalNodes;      
    }
    
    GhostPseudoMesh* Mesh3D::_createGhostPseudoMesh(unsigned ghostLayerDepth) {
        auto ghostNodesPerDirection = _createNumberOfGhostNodesPerDirectionMap(ghostLayerDepth);

        auto ghostNodesList = make_shared<list <Node*>>(0);

        // Parametric coordinate 1 of nodes in the new ghost mesh
        auto nodeArrayPositionI = 0;
        // Parametric coordinate 2 of nodes in the new ghost mesh
        auto nodeArrayPositionJ = 0;
        // Parametric coordinate 3 of nodes in the new ghost mesh
        auto nodeArrayPositionK = 0;
        
        auto nn1 = nodesPerDirection[One];
        auto nn1Ghost = ghostNodesPerDirection->at(One);
        auto nn2 = nodesPerDirection[Two];
        auto nn2Ghost = ghostNodesPerDirection->at(Two);
        auto nn3 = nodesPerDirection[Three];
        auto nn3Ghost = ghostNodesPerDirection->at(Three);

        auto parametricCoordToNodeMap =  createParametricCoordToNodesMap();
        
        for (int k = -static_cast<int>(nn3Ghost); k < static_cast<int>(nn3) + static_cast<int>(nn3Ghost); k++){
            for (int j = -static_cast<int>(nn2Ghost); j < static_cast<int>(nn2) + static_cast<int>(nn2Ghost); j++) {
                for (int i = -static_cast<int>(nn1Ghost); i < static_cast<int>(nn1) + static_cast<int>(nn1Ghost); ++i) {
                    auto parametricCoords = vector<double>{static_cast<double>(i), static_cast<double>(j),
                                                           static_cast<double>(k)};
                    // If node is inside the original mesh add it to the ghost mesh Array
                    if (parametricCoordToNodeMap->find(parametricCoords) == parametricCoordToNodeMap->end()) {
                        auto node = new Node();
                        node->coordinates.setPositionVector(make_shared<vector<double>>(parametricCoords), Parametric);
                        vector<double> templateCoord = {static_cast<double>(i) * specs->templateStepOne,
                                                        static_cast<double>(j) * specs->templateStepTwo};
                        // Rotate 
                        Transformations::rotate(templateCoord, specs->templateRotAngleOne);
                        // Shear
                        Transformations::shear(templateCoord, specs->templateShearOne, specs->templateShearTwo);

                        node->coordinates.setPositionVector(make_shared<vector<double>>(templateCoord), Template);
                        ghostNodesList->push_back(node);
                    }
                    nodeArrayPositionI++;
                }
                nodeArrayPositionI = 0;
                nodeArrayPositionJ++;
            }
            nodeArrayPositionJ = 0;
            nodeArrayPositionK++;
        }
        return new GhostPseudoMesh(ghostNodesList, ghostNodesPerDirection, parametricCoordToNodeMap);
    }
    
    shared_ptr<map<vector<double>, Node*>> Mesh3D::createParametricCoordToNodesMap() {
        auto parametricCoordToNodeMap = make_shared<map<vector<double>, Node*>>();
        for (auto& node : *totalNodesVector) {
            auto parametricCoords = node->coordinates.positionVector(Parametric);
            parametricCoordToNodeMap->insert(pair<vector<double>, Node*>(parametricCoords, node));
        }
        return parametricCoordToNodeMap;
    }

    vector<double> Mesh3D::getNormalUnitVectorOfBoundaryNode(Position boundaryPosition, Node *node) {
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

            vector<double> covariantBaseVector1 = metrics->at(*node->id.global)->covariantBaseVectors->at(direction1);
            vector<double> covariantBaseVector2 = metrics->at(*node->id.global)->covariantBaseVectors->at(direction2);
            
            vector<double> normalUnitVector = VectorOperations::crossProduct(covariantBaseVector1, covariantBaseVector2);
            VectorOperations::normalize(normalUnitVector);
            
            cout<<*node->id.global<<endl;
            cout<<boundaryPosition<<" "<<normalUnitVector[0]<<" "<<normalUnitVector[1]<<" "<<normalUnitVector[2]<<endl;
            
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

/*    void Mesh3D::storeMeshInVTKFile(const string &filePath, const string &fileName, CoordinateType coordinateType,
                                    bool StoreOnlyNodes) const {
        ofstream outputFile(filePath + fileName);
        outputFile << "# vtk DataFile Version 3.0 \n";
        outputFile << "vtk output \n";
        outputFile << "ASCII \n";
        outputFile << "DATASET UNSTRUCTURED_GRID \n";
        outputFile << "POINTS " << totalNodesVector->size() << " double\n";
        for (auto &node: *totalNodesVector) {
            auto coordinates = node->coordinates.positionVector3D(coordinateType);
            outputFile << coordinates[0] << " " << coordinates[1] << " " << coordinates[2] << "\n";
        }

        if (!StoreOnlyNodes) {
            if (elements == nullptr) {
                throw runtime_error("Elements not created yet.");
            }
            vector<unsigned int> connectivityList;
            unsigned int nodesPerElement = 0;
            switch (elements->elementType()) {
                case Hexahedron:
                    connectivityList = {0, 1, 2, 3, 5, 4, 7, 6};
                    nodesPerElement = 8;
                    break;
                case Wedge:
                    // TODO: Define the connectivity list for the Wedge
                    break;
                default:
                    throw runtime_error("3D geometry only supports Hexahedron and Wedge elements.");
            }

            unsigned int totalIntsForCells = elements->numberOfElements() * (1 + nodesPerElement);
            outputFile << "CELLS " << elements->numberOfElements() << " " << totalIntsForCells << "\n";

            for (unsigned int i = 0; i < elements->numberOfElements(); ++i) {
                outputFile << nodesPerElement << " ";
                for (auto index: connectivityList) {
                    outputFile << elements->getElement(i)->nodes()->at(index)->id.global << " ";
                }
                outputFile << "\n";
            }

            // Adding CELL_TYPES section
            outputFile << "CELL_TYPES " << elements->numberOfElements() << "\n";
            for (unsigned int i = 0; i < elements->numberOfElements(); ++i) {
                if (elements->elementType() == Hexahedron) {
                    outputFile << "12\n"; // VTK's ID for hexahedron
                } else if (elements->elementType() == Wedge) {
                    outputFile << "13\n"; // VTK's ID for wedge
                }
            }
        }

        outputFile.close();
    }*/

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
            auto coordinates = node->coordinates.positionVector3D(coordinateType);
            outputFile << coordinates[0] << " " << coordinates[1] << " " << coordinates[2] << "\n";
        }

        outputFile.close();
    }



} // Discretization