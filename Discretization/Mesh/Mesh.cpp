//
// Created by hal9000 on 12/17/22.
//

#include "Mesh.h"

using namespace  Discretization;

namespace Discretization {

    Mesh::Mesh(shared_ptr<Array<Node*>>nodes) {
        _nodesMatrix = std::move(nodes);
        isInitialized = false;
        elements = nullptr;
        logs = Logs("Mesh");
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
    
    Node *Mesh::nodeFromID(unsigned* ID) {
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
    
    void Mesh::printMesh() {}

    NumericalVector<double> Mesh::getNormalUnitVectorOfBoundaryNode(Position boundaryPosition, Node *node) {
        return {};
    }
    
    unique_ptr<vector<Node*>> Mesh::getInternalNodesVector() {
        return nullptr;
    }

    void Mesh::initialize() {
        isInitialized = true;
        nodesPerDirection.insert(make_pair(One, _nodesMatrix->numberOfRows()));
        nodesPerDirection.insert(make_pair(Two, _nodesMatrix->numberOfColumns()));
        nodesPerDirection.insert(make_pair(Three, _nodesMatrix->numberOfAisles()));
        _addDBoundaryNodesToMap();
        _addTotalNodesToVector();
        _createNodeToIdMap();
    }
    
    unique_ptr<vector<Node*>> Mesh::getBoundaryNodesVector(){
        auto boundaryNodesList = list<Node*>();
        for (auto &boundaryNodesMap : *boundaryNodes)
            for (auto &node : *boundaryNodesMap.second)
                if (find(boundaryNodesList.begin(), boundaryNodesList.end(), node) == boundaryNodesList.end())
                    boundaryNodesList.push_back(node);
        return make_unique<vector<Node*>>(boundaryNodesList.begin(), boundaryNodesList.end());
    }

    void Mesh::_cleanMeshDataStructures() {
        //search all boundaryNodes unordered_map and deallocate all vector pointer values
        for (auto &node : *totalNodesVector) {
            delete node;
            node = nullptr;
        }
        totalNodesVector = nullptr;
        boundaryNodes = nullptr;
        _nodesMatrix = nullptr;
    }

    void Mesh::calculateMeshMetrics(CoordinateType coordinateSystem, bool isUniformMesh) {
        metrics = make_shared<unordered_map<Node*, shared_ptr<Metrics>>>();
        cout << "Calculating mesh metrics..." << endl;
        _arbitrarilySpacedMeshMetrics(coordinateSystem, getBoundaryNodesVector(), true);
        _arbitrarilySpacedMeshMetrics(coordinateSystem, getInternalNodesVector(), false);
        cout << "Finished calculating mesh metrics" << endl;
    }
    

    void Mesh::_arbitrarilySpacedMeshMetrics(CoordinateType coordinateSystem, unique_ptr<vector<Discretization::Node *>> nodes, bool areBoundary) {

        using findColinearNodes = map<Direction, shared_ptr<NumericalVector<double>>> (IsoParametricNodeGraph::*)(
                CoordinateType, map<Position, vector<Node *>> &) const;
        findColinearNodes colinearNodes;

        if (areBoundary)
            colinearNodes = &IsoParametricNodeGraph::getSameColinearNodalCoordinatesOnBoundary;
        else
            colinearNodes = &IsoParametricNodeGraph::getSameColinearNodalCoordinates;

        auto start = std::chrono::steady_clock::now(); // Start the timer
        auto schemeSpecs = make_shared<FiniteDifferenceSchemeOrder>(2, directions());
        auto directions = this->directions();
        short unsigned maxDerivativeOrder = 1;
        auto parametricCoordsMap = getCoordinatesToNodesMap(Parametric);

        auto schemeBuilder = FiniteDifferenceSchemeBuilder(schemeSpecs);

        auto errorOrderDerivative1 = schemeSpecs->getErrorForDerivativeOfArbitraryScheme(1);

        map<short unsigned, map<Direction, map<vector<Position>, short>>> templatePositionsAndPointsMap = schemeBuilder.initiatePositionsAndPointsMap(
                maxDerivativeOrder, directions);
        schemeBuilder.templatePositionsAndPoints(1, errorOrderDerivative1, directions,
                                                 templatePositionsAndPointsMap[1]);
        auto maxNeighbours = schemeBuilder.getMaximumNumberOfPointsForArbitrarySchemeType();
        for (auto node: *nodes)
            metrics->insert(make_pair(node, make_shared<Metrics>(node, directions.size())));

        auto nodalMetricsCalculationJob = [&](unsigned startNodeIndex, unsigned endNodeIndex) -> void {

            for (auto nodeIndex = startNodeIndex; nodeIndex < endNodeIndex; nodeIndex++) {
                Node *node = (*nodes)[nodeIndex];
                auto graph = IsoParametricNodeGraph(node, maxNeighbours, parametricCoordsMap, nodesPerDirection,
                                                    false);
                auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);
                auto nodeMetrics = metrics->at(node);
                //Loop through all the directions to find g_i = d(x_j)/d(x_i), g^i = d(x_i)/d(x_j)
                for (auto &directionI: directions) {
                    auto i = spatialDirectionToUnsigned[directionI];
                    auto covariantBaseVectorI = NumericalVector<double>(directions.size(), 0);
                    auto contravariantBaseVectorI = NumericalVector<double>(directions.size(), 0);

                    for (auto &directionJ: directions) {
                        auto j = spatialDirectionToUnsigned[directionJ];

                        //Check if the available positions are qualified for the current derivative order
                        auto qualifiedPositions = schemeBuilder.getQualifiedFromAvailable(
                                availablePositionsAndDepth[directionJ],
                                templatePositionsAndPointsMap[1][directionJ]);

                        auto graphFilter = map<Position, unsigned short>();
                        for (auto &tuple: qualifiedPositions) {
                            for (auto &point: tuple.first) {
                                graphFilter.insert(pair<Position, unsigned short>(point, tuple.second));
                            }
                        }
                        auto filteredNodeGraph = graph.getNodeGraph(graphFilter);

                        auto parametricCoords = (graph.*colinearNodes)(Parametric, filteredNodeGraph);
                        auto templateCoords = (graph.*colinearNodes)(coordinateSystem, filteredNodeGraph);

                        auto taylorPointParametric = (*node->coordinates.getPositionVector(Parametric))[i];
                        auto covariantWeights = calculateWeightsOfDerivativeOrder(
                                *parametricCoords[directionJ]->getVectorSharedPtr(), 1, taylorPointParametric);

                        auto taylorPointTemplate = (*node->coordinates.getPositionVector(coordinateSystem))[i];
                        auto contravariantWeights = calculateWeightsOfDerivativeOrder(
                                *templateCoords[directionJ]->getVectorSharedPtr(), 1, taylorPointTemplate);


                        //Check if the number of weights and the number of nodes match
                        if (covariantWeights.size() != parametricCoords[directionJ]->size()) {
                            throw std::runtime_error(
                                    "Number of weights and number of template nodal coords do not match"
                                    " for node " + to_string(*node->id.global) +
                                    " in direction " + to_string(directionI) +
                                    " Cannot calculate covariant base vectors");
                        }

                        if (contravariantWeights.size() != templateCoords[directionJ]->size()) {
                            throw std::runtime_error(
                                    "Number of weights and number of parametric nodal coords do not match"
                                    " for node " + to_string(*node->id.global) +
                                    " in direction " + to_string(directionI) +
                                    " Cannot calculate contravariant base vectors");
                        }

                        //Covariant base vectors (dr_i/dξ_i)
                        //g_1 = {dx/dξ, dy/dξ, dz/dξ}
                        //g_2 = {dx/dη, dy/dη, dz/dη} 
                        //g_3 = {dx/dζ, dy/dζ, dz/dζ}
                        covariantBaseVectorI[i] = covariantWeights.dotProduct(templateCoords[directionJ]);
                        //Contravariant base vectors (dξ_i/dr_i)
                        //g^1 = {dξ/dx, dξ/dy, dξ/dz}
                        //g^2 = {dη/dx, dη/dy, dη/dz}
                        //g^3 = {dζ/dx, dζ/dy, dζ/dz}
                        contravariantBaseVectorI[i] = contravariantWeights.dotProduct(parametricCoords[directionJ]);
                    }
                    nodeMetrics->covariantBaseVectors->insert(
                            pair<Direction, NumericalVector<double>>(directionI, covariantBaseVectorI));
                    nodeMetrics->contravariantBaseVectors->insert(
                            pair<Direction, NumericalVector<double>>(directionI, contravariantBaseVectorI));
                }
                nodeMetrics->calculateCovariantTensor();
                nodeMetrics->calculateContravariantTensor();
            }
        };
        ThreadingOperations<double>::executeParallelJob(nodalMetricsCalculationJob, nodes->size(), 12);
    }
    

    shared_ptr<map<vector<double>, Node*>> Mesh::getCoordinatesToNodesMap(CoordinateType coordinateType) {
        auto parametricCoordToNodeMap = make_shared<map<vector<double>, Node*>>();
        for (auto& node : *totalNodesVector) {
            auto nodalParametricCoordsSTD = *node->coordinates.getPositionVector3D(coordinateType).getVectorSharedPtr();
            parametricCoordToNodeMap->insert(pair<vector<double>, Node*>(std::move(nodalParametricCoordsSTD), node));
        }
        return parametricCoordToNodeMap;
    }

    unique_ptr<unordered_map<Node *, Position>> Mesh::getBoundaryNodeToPositionMap() const {
        auto boundaryNodeToPositionMap = make_unique<unordered_map<Node *, Position>>();
        for (auto &boundary : *boundaryNodes)
            for (auto &node : *boundary.second)
                boundaryNodeToPositionMap->insert(pair<Node *, Position>(node, boundary.first));
        return boundaryNodeToPositionMap;
    }
    
    unsigned Mesh::numberOfInternalNodes() {
        return 0;
    }

    void Mesh::createElements(ElementType elementType, unsigned int nodesPerEdge) {

    }

    void Mesh::storeMeshInVTKFile(const string &filePath, const string &fileName, CoordinateType coordinateType,
                                  bool StoreOnlyNodes) const {

    }


    void Mesh::_addDBoundaryNodesToMap() {
    }

    void Mesh::_addTotalNodesToVector() {
    }
    
    void Mesh::_createNodeToIdMap() {
        _nodesMap = make_unique<unordered_map<unsigned*, Node*>>();
        for (auto &node : *totalNodesVector)
            _nodesMap->insert(make_pair(node->id.global, node));
    }

} // Discretization