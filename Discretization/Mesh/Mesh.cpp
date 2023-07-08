//
// Created by hal9000 on 12/17/22.
//

#include "Mesh.h"
#include "../../LinearAlgebra/FiniteDifferences/FDSchemeType.h"
#include "../../LinearAlgebra/FiniteDifferences/FDSchemeSpecs.h"
#include "../../Utility/Exporters/Exporters.h"

using namespace  Discretization;

namespace Discretization {

    Mesh::Mesh() {
        isInitialized = false;
        _nodesMatrix = nullptr;
        boundaryNodes = nullptr;
        totalNodesVector = nullptr;
        _nodesMap = nullptr;

    }

    Mesh::~Mesh() {
        _nodesMatrix = nullptr;
        boundaryNodes = nullptr;
        totalNodesVector = nullptr;
    }

    unsigned Mesh::numberOfTotalNodes() {
        if (isInitialized)
           return _nodesMatrix->size();
        else
            throw std::runtime_error("Mesh has not been initialized");
    }
    
    unsigned Mesh::numberOfBoundaryNodes() {
        return numberOfTotalNodes() - numberOfInternalNodes();
    }
    
    Node *Mesh::nodeFromID(unsigned ID) {
        if (isInitialized)
            return _nodesMap->at(ID);
        else
            return nullptr;
    }
    
    unsigned Mesh::dimensions() { return 0; }

    SpaceEntityType Mesh::space() {
        return NullSpace;
    }

    vector<Direction> Mesh::directions() {
        return {};
    }

    Node *Mesh::node(unsigned i) {
        return nullptr;
    }

    Node *Mesh::node(unsigned i, unsigned j) {
        return nullptr;
    }

    Node *Mesh::node(unsigned i, unsigned j, unsigned k) {
        return nullptr;
    }

    shared_ptr<map<vector<double>, Node *>>Mesh::createParametricCoordToNodesMap() {
        return nullptr;
    }

    void Mesh::printMesh() {}
    
    
    unique_ptr<vector<Node*>> Mesh::getInternalNodesVector() {
        return nullptr;
    }

    void Mesh::initialize() {
        isInitialized = true;
        _createNumberOfNodesPerDirectionMap();
        _categorizeNodes();
    }

    shared_ptr<map<Position, shared_ptr<vector<Node*>>>> Mesh::_addDBoundaryNodesToMap() {
        return nullptr;
    }


    shared_ptr<vector<Node *>>Mesh::_addTotalNodesToVector() {
        return nullptr;
    }
    
    unique_ptr<vector<Node*>> Mesh::getBoundaryNodesVector(){
        auto boundaryNodesList = list<Node*>();
        for (auto &boundaryNodesMap : *boundaryNodes)
            for (auto &node : *boundaryNodesMap.second)
                if (find(boundaryNodesList.begin(), boundaryNodesList.end(), node) == boundaryNodesList.end())
                    boundaryNodesList.push_back(node);
        return make_unique<vector<Node*>>(boundaryNodesList.begin(), boundaryNodesList.end());
    }
    
    void Mesh::_categorizeNodes() {
        if (isInitialized) {
            boundaryNodes = _addDBoundaryNodesToMap();
            totalNodesVector = _addTotalNodesToVector();
        } else
            throw std::runtime_error("Mesh has not been initialized");
    }

    void Mesh::_createNumberOfNodesPerDirectionMap() {
        if (isInitialized) {
            nodesPerDirection.insert(pair<Direction, unsigned>(One, _nodesMatrix->numberOfRows()));
            nodesPerDirection.insert(pair<Direction, unsigned>(Two, _nodesMatrix->numberOfColumns()));
            nodesPerDirection.insert(pair<Direction, unsigned>(Three, _nodesMatrix->numberOfAisles()));
        } else
            throw std::runtime_error("Mesh has not been initialized");
    }

    map<unsigned, Node *> *Mesh::_createNodesMap() const {
        if (isInitialized) {
            auto nodesMap = new map<unsigned, Node *>();
            for (auto &node: *totalNodesVector) {

                nodesMap->insert(pair<unsigned, Node *>(*node->id.global, node));
            }
            return nodesMap;
        } else
            throw std::runtime_error("Mesh has not been initialized");
    }

    void Mesh::_cleanMeshDataStructures() {
        //search all boundaryNodes map and deallocate all vector pointer values
        for (auto &node : *totalNodesVector) {
            delete node;
            node = nullptr;
        }
        totalNodesVector = nullptr;
        boundaryNodes = nullptr;
        delete _nodesMap;
        _nodesMap = nullptr;
    }

    shared_ptr<map<Direction, unsigned>> Mesh::_createNumberOfGhostNodesPerDirectionMap(unsigned ghostLayerDepth) {
        auto numberOfGhostNodesPerDirection = make_shared<map<Direction, unsigned>>();
        for (auto &direction: directions()) {
            numberOfGhostNodesPerDirection->insert(pair<Direction, unsigned>(direction, ghostLayerDepth));
        }
        return numberOfGhostNodesPerDirection;
    }

    void Mesh::calculateMeshMetrics(CoordinateType coordinateSystem, bool isUniformMesh) {
        metrics = make_shared<map<unsigned, shared_ptr<Metrics>>>();

        if (isUniformMesh) {
            _uniformlySpacedMetrics(coordinateSystem, getBoundaryNodesVector(), true);
            _uniformlySpacedMetrics(coordinateSystem, getInternalNodesVector(), false);
        } else {
            _arbitrarilySpacedMeshMetrics(coordinateSystem);
        }
        
/*        for (auto &node: *metrics) {
            //node.first->printNode();
            cout << "Node " << node.first << endl;
            cout << "covariant tensor" << endl;
            node.second->covariantTensor->print();
            cout << "contravariant tensor" << endl;
            node.second->contravariantTensor->print();
            cout << "---------------------------------" << endl;
        }*/
    }

    GhostPseudoMesh* Mesh::_createGhostPseudoMesh(unsigned ghostLayerDepth) {
        return nullptr;
    }
    
    void Mesh::_arbitrarilySpacedMeshMetrics(CoordinateType coordinateSystem) {
        if (isInitialized) {
            //Initialize Mesh Metrics map
            metrics = make_shared<map<unsigned, shared_ptr<Metrics>>>();

            //Create Scheme Specs. Metrics are calculated by a central ("diamond") scheme
            auto schemeSpecs = make_shared<FDSchemeSpecs>(Central, specs->metricsOrder, directions());
            //Create Scheme Builder to gain access to utility functions for the scheme creation
            auto schemeBuilder = new FiniteDifferenceSchemeBuilder(schemeSpecs);
            // Initiate GhostPseudoMesh
            auto ghostMesh = _createGhostPseudoMesh(schemeBuilder->getNumberOfGhostNodesNeeded());
            
            //March through all the nodes of the mesh and calculate its metrics
            for (auto &node: *totalNodesVector) {
                //Find Neighbors in a manner that applies the same numerical scheme to all nodes
                auto neighbours = schemeBuilder->getNumberOfDiagonalNeighboursNeeded();
                //Initiate Node Graph
                auto graph = new IsoParametricNodeGraph(node, schemeBuilder->getNumberOfGhostNodesNeeded(), ghostMesh->parametricCoordToNodeMap,
                                                        nodesPerDirection, false);

                //Get the adjusted node graph that contains only the nodes that are needed to calculate the FD scheme
                auto nodeGraph = graph->getNodeGraph(neighbours);

                //Initialize Metrics class for the current node.
                auto nodeMetrics = new Metrics(node, dimensions());

                //Get the co-linear nodal coordinates (Left-Right -> 1, Up-Down -> 2, Front-Back -> 3)
                auto parametricCoords = graph->getSameColinearNodalCoordinates(Parametric);
                auto templateCoords = graph->getSameColinearNodalCoordinates(coordinateSystem);


                auto directionsVector = directions();
                //March through all the directions I (g_i = d(x_j)/d(x_i))
                for (auto&  directionI : directionsVector){//Initialize the weights vector. Their values depend on whether the mesh is uniform or not.

                    auto i = spatialDirectionToUnsigned[directionI];
                    
                    auto covariantBaseVectorI = vector<double>(directionsVector.size(), 0);
                    auto contravariantBaseVectorI = vector<double>(directionsVector.size(), 0);
                    
                    auto covariantWeights = calculateWeightsOfDerivativeOrder(
                            parametricCoords[directionI][i], 1, node->coordinates(Parametric, i));
                    auto contravariantWeights = calculateWeightsOfDerivativeOrder(
                            templateCoords[directionI][i], 1, node->coordinates(coordinateSystem, i));
                    
                    for (auto &directionJ: directionsVector){
                        auto j = spatialDirectionToUnsigned[directionJ];

                        //Covariant base vectors (dr_i/dξ_i)
                        //g_1 = {dx/dξ, dy/dξ, dz/dξ}
                        //g_2 = {dx/dη, dy/dη, dz/dη}
                        //g_3 = {dx/dζ, dy/dζ, dz/dζ}
                        //auto gi = VectorOperations::dotProduct(covariantWeights, templateCoordsMap[directionJ]);
                        covariantBaseVectorI[j] = VectorOperations::dotProduct(covariantWeights, templateCoords[directionI][j]);

                        //Contravariant base vectors (dξ_i/dr_i)
                        //g^1 = {dξ/dx, dξ/dy, dξ/dz}
                        //g^2 = {dη/dx, dη/dy, dη/dz}
                        //g^3 = {dζ/dx, dζ/dy, dζ/dz}
                        contravariantBaseVectorI[j] = VectorOperations::dotProduct(contravariantWeights, parametricCoords[directionI][j]);
                    }
                    nodeMetrics->covariantBaseVectors->insert(pair<Direction, vector<double>>(directionI, covariantBaseVectorI));
                    nodeMetrics->contravariantBaseVectors->insert(pair<Direction, vector<double>>(directionI, contravariantBaseVectorI));
                }

                nodeMetrics->calculateCovariantTensor();
                nodeMetrics->calculateContravariantTensor();
                metrics->insert(pair<unsigned, shared_ptr<Metrics> >(*node->id.global, nodeMetrics));

                //Deallocate memory
            }
            delete ghostMesh;
            delete schemeBuilder;
            schemeBuilder = nullptr;
            schemeSpecs = nullptr;
        }
        else
            throw std::runtime_error("Mesh is not initialized");
    }

    void Mesh::_uniformlySpacedMetrics(CoordinateType coordinateSystem, unique_ptr<vector<Discretization::Node *>> nodes, bool areBoundary) {
        
        using findColinearNodes = map<Direction, vector<vector<double>>> (IsoParametricNodeGraph::*)(CoordinateType, map<Position, vector<Node*>>& ) const;
        findColinearNodes colinearNodes;
        
        if (areBoundary)
            colinearNodes = &IsoParametricNodeGraph::getSameColinearNodalCoordinatesOnBoundary;
        else
            colinearNodes = &IsoParametricNodeGraph::getSameColinearNodalCoordinates;
        
        auto start = std::chrono::steady_clock::now(); // Start the timer
        auto schemeSpecs = make_shared<FDSchemeSpecs>(2, directions());
        auto directions = this->directions();
        short unsigned maxDerivativeOrder = 1;
        auto parametricCoordsMap = this->createParametricCoordToNodesMap();

        auto schemeBuilder = FiniteDifferenceSchemeBuilder(schemeSpecs);
        
        auto errorOrderDerivative1 = schemeSpecs->getErrorForDerivativeOfArbitraryScheme(1);
        
        map<short unsigned, map<Direction, map<vector<Position>, short>>> templatePositionsAndPointsMap = schemeBuilder.initiatePositionsAndPointsMap(maxDerivativeOrder, directions);
        schemeBuilder.templatePositionsAndPoints(1, errorOrderDerivative1, directions, templatePositionsAndPointsMap[1]);
        auto maxNeighbours = schemeBuilder.getMaximumNumberOfPointsForArbitrarySchemeType();

        for (auto &node: *nodes) {

            auto graph = IsoParametricNodeGraph(node, maxNeighbours, parametricCoordsMap, nodesPerDirection, false);
            auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);
            auto nodeMetrics = make_shared<Metrics>(node, dimensions());

            for (auto &directionI: directions) {
                auto directionIndex = spatialDirectionToUnsigned[directionI];

                //Check if the available positions are qualified for the current derivative order
                auto qualifiedPositions = schemeBuilder.getQualifiedFromAvailable(
                        availablePositionsAndDepth[directionI],
                        templatePositionsAndPointsMap[1][directionI]);
                auto scheme = FiniteDifferenceSchemeBuilder::getSchemeWeightsFromQualifiedPositions(
                        qualifiedPositions, directionI, errorOrderDerivative1, 1);

                auto graphFilter = map<Position, unsigned short>();
                for (auto &tuple: qualifiedPositions) {
                    for (auto &point: tuple.first) {
                        graphFilter.insert(pair<Position, unsigned short>(point, tuple.second));
                    }
                }
                auto filteredNodeGraph = graph.getNodeGraph(graphFilter);

                auto parametricCoords = (graph.*colinearNodes)(Parametric, filteredNodeGraph);
                auto templateCoords = (graph.*colinearNodes)(coordinateSystem, filteredNodeGraph);

                auto i = spatialDirectionToUnsigned[directionI];

                auto covariantBaseVectorI = vector<double>(directions.size(), 0);
                auto contravariantBaseVectorI = vector<double>(directions.size(), 0);

                //Get the FD scheme weights for the current direction
                auto covariantWeights = scheme.weights;
                auto contravariantWeights = covariantWeights;

                //Check if the number of weights and the number of nodes match
                if (covariantWeights.size() != parametricCoords[directionI][i].size()) {
                    throw std::runtime_error(
                            "Number of weights and number of template nodal coords do not match"
                            " for node " + to_string(*node->id.global) +
                            " in direction " + to_string(directionI) +
                            " Cannot calculate covariant base vectors");
                }

                if (contravariantWeights.size() != templateCoords[directionI][i].size()) {
                    throw std::runtime_error(
                            "Number of weights and number of parametric nodal coords do not match"
                            " for node " + to_string(*node->id.global) +
                            " in direction " + to_string(directionI) +
                            " Cannot calculate contravariant base vectors");
                }

                auto covariantStep = 1.0;
                auto contravariantStep = VectorOperations::averageAbsoluteDifference(templateCoords[directionI][i]);
                contravariantStep = pow(contravariantStep, scheme.power) * scheme.denominatorCoefficient;

                for (int weight = 0; weight < covariantWeights.size(); weight++) {
                    covariantWeights[weight] /= covariantStep;
                    contravariantWeights[weight] /= contravariantStep;
                }
                for (auto &directionJ: directions) {
                    auto j = spatialDirectionToUnsigned[directionJ];

                    //Covariant base vectors (dr_i/dξ_i)
                    //g_1 = {dx/dξ, dy/dξ, dz/dξ}
                    //g_2 = {dx/dη, dy/dη, dz/dη}
                    //g_3 = {dx/dζ, dy/dζ, dz/dζ}
                    //auto gi = VectorOperations::dotProduct(covariantWeights, templateCoordsMap[directionJ]);
                    covariantBaseVectorI[j] = VectorOperations::dotProduct(covariantWeights,templateCoords[directionI][j]);
                    //Contravariant base vectors (dξ_i/dr_i)
                    //g^1 = {dξ/dx, dξ/dy, dξ/dz}
                    //g^2 = {dη/dx, dη/dy, dη/dz}
                    //g^3 = {dζ/dx, dζ/dy, dζ/dz}
                    contravariantBaseVectorI[j] = VectorOperations::dotProduct(contravariantWeights,parametricCoords[directionI][j]);
                }
                nodeMetrics->covariantBaseVectors->insert(
                        pair<Direction, vector<double>>(directionI, covariantBaseVectorI));
                nodeMetrics->contravariantBaseVectors->insert(
                        pair<Direction, vector<double>>(directionI, contravariantBaseVectorI));
            }
            nodeMetrics->calculateCovariantTensor();
            nodeMetrics->calculateContravariantTensor();
            metrics->insert(pair<unsigned, shared_ptr<Metrics> >(*node->id.global, nodeMetrics));
/*            cout << "Node " << *node->id.global << " calculated" << endl;
            nodeMetrics->covariantTensor->print();*/
        }
    }
    
    
    void Mesh::storeMeshInVTKFile(const std::string& filePath, const std::string& fileName, CoordinateType coordinateType) const {
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
        outputFile.close();
    }

/*    void Mesh::storeMeshInVTKFile(const std::string& filePath, const std::string& fileName, CoordinateType coordinateType) const {
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

        // Assume a 2D grid, so each cell is a quadrilateral made of 4 points
        unsigned int numCells = totalNodesVector->size() - nodesPerDirection.at(One) - 1;
        outputFile << "CELLS " << numCells << " " << numCells * 5 << "\n";
        for (unsigned int i = 0; i < totalNodesVector->size() - nodesPerDirection.at(One) - 1; i++) {
            outputFile << 4 << " " << i << " " << i + 1 << " " << i + nodesPerDirection.at(One) + 1 << " " << i + nodesPerDirection.at(One) << "\n";
        }

        // Specify cell type. For a 2D grid, cells are quadrilaterals, which have cell type 9 in VTK.
        outputFile << "CELL_TYPES " << numCells << "\n";
        for (unsigned int i = 0; i < numCells; i++) {
            outputFile << 9 << "\n";
        }

        outputFile.close();
    }*/

    map<vector<double>, Node *> Mesh::getCoordinateToNodeMap(CoordinateType coordinateType) const {
        map<vector<double>, Node *> coordinateToNodeMap;
        for (auto &node: *totalNodesVector) {
            coordinateToNodeMap.insert(pair<vector<double>, Node *>(node->coordinates.positionVector3D(coordinateType), node));
        }
        return coordinateToNodeMap;
    }

    unsigned Mesh::numberOfInternalNodes() {
        return 0;
    }


} // Discretization