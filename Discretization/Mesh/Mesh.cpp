//
// Created by hal9000 on 12/17/22.
//

#include "Mesh.h"
#include "../../Utility/Exporters/Exporters.h"

using namespace  Discretization;

namespace Discretization {

    Mesh::Mesh() {
        isInitialized = false;
        _nodesMatrix = nullptr;
        boundaryNodes = nullptr;
        totalNodesVector = nullptr;
        _nodesMap = nullptr;
        elements = nullptr;

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
    
    void Mesh::printMesh() {}

    NumericalVector<double> Mesh::getNormalUnitVectorOfBoundaryNode(Position boundaryPosition, Node *node) {
        return {};
    }
    
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
            //_arbitrarilySpacedMeshMetrics(coordinateSystem);
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

/*    GhostPseudoMesh* Mesh::_createGhostPseudoMesh(unsigned ghostLayerDepth) {
        return nullptr;
    }*/
    
/*    void Mesh::_arbitrarilySpacedMeshMetrics(CoordinateType coordinateSystem) {
        if (isInitialized) {
            //Initialize Mesh Metrics map
            metrics = make_shared<map<unsigned, shared_ptr<Metrics>>>();

            //Create Scheme Specs. Metrics are calculated by a central ("diamond") scheme
            auto schemeSpecs = make_shared<FDSchemeSpecs>(Central, specs->metricsOrder, directions());
            //Create Scheme Builder to gain access to utility functions for the scheme creation
            auto schemeBuilder = new FiniteDifferenceSchemeBuilder(schemeSpecs);
            // Initiate GhostPseudoMesh
            //auto ghostMesh = _createGhostPseudoMesh(schemeBuilder->getNumberOfGhostNodesNeeded());
            
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
                    
                    auto covariantBaseVectorI = NumericalVector<double>(directionsVector.size(), 0);
                    auto contravariantBaseVectorI = NumericalVector<double>(directionsVector.size(), 0);
                    
                    auto covariantWeights = calculateWeightsOfDerivativeOrder(
                            parametricCoords[directionI][i], 1, node->coordinates(Parametric, i));
                    auto contravariantWeights = calculateWeightsOfDerivativeOrder(
                            templateCoords[directionI][i], 1, node->coordinates(coordinateSystem, i));
                    
                    for (auto &directionJ: directionsVector){
                        auto j = spatialDirectionToUnsigned[directionJ];
                        auto coords = NumericalVector<double>(templateCoords[directionI][j].size());
                        for (unsigned k=0; k<coords.size(); ++k){
                            coords[k] = templateCoords[directionI][j][k];
                        }
                        //Covariant base vectors (dr_i/dξ_i)
                        //g_1 = {dx/dξ, dy/dξ, dz/dξ}
                        //g_2 = {dx/dη, dy/dη, dz/dη}
                        //g_3 = {dx/dζ, dy/dζ, dz/dζ}
                        //auto gi = VectorOperations::dotProduct(covariantWeights, templateCoordsMap[directionJ]);
                        covariantBaseVectorI[j] = covariantWeights.dotProduct(coords);

                        for (unsigned k=0; k<coords.size(); ++k){
                            coords[k] = parametricCoords[directionI][j][k];
                        }
                        //Contravariant base vectors (dξ_i/dr_i)
                        //g^1 = {dξ/dx, dξ/dy, dξ/dz}
                        //g^2 = {dη/dx, dη/dy, dη/dz}
                        //g^3 = {dζ/dx, dζ/dy, dζ/dz}
                        contravariantBaseVectorI[j] = contravariantWeights.dotProduct(coords);
                        
                    }
                    nodeMetrics->covariantBaseVectors->insert(pair<Direction, NumericalVector<double>>(directionI, covariantBaseVectorI));
                    nodeMetrics->contravariantBaseVectors->insert(pair<Direction, NumericalVector<double>>(directionI, contravariantBaseVectorI));
                }

                nodeMetrics->calculateCovariantTensor();
                nodeMetrics->calculateContravariantTensor();
                metrics->insert(pair<unsigned, shared_ptr<Metrics> >(*node->id.global, nodeMetrics));

            }
            delete ghostMesh;
            delete schemeBuilder;
            schemeBuilder = nullptr;
            schemeSpecs = nullptr;
        }
        else
            throw std::runtime_error("Mesh is not initialized");
    }*/

    void Mesh::_uniformlySpacedMetrics(CoordinateType coordinateSystem, unique_ptr<vector<Discretization::Node *>> nodes, bool areBoundary) {

        using findColinearNodes = map<Direction, vector<shared_ptr<NumericalVector<double>>>> (IsoParametricNodeGraph::*)(CoordinateType, map<Position, vector<Node*>>& ) const;
        findColinearNodes colinearNodes;

        if (areBoundary)
            colinearNodes = &IsoParametricNodeGraph::getSameColinearNodalCoordinatesOnBoundary;
        else
            colinearNodes = &IsoParametricNodeGraph::getSameColinearNodalCoordinates;

        auto start = std::chrono::steady_clock::now(); // Start the timer
        auto schemeSpecs = make_shared<FDSchemeSpecs>(2, directions());
        auto directions = this->directions();
        short unsigned maxDerivativeOrder = 1;
        auto parametricCoordsMap = getCoordinatesToNodesMap(Parametric);

        auto schemeBuilder = FiniteDifferenceSchemeBuilder(schemeSpecs);

        auto errorOrderDerivative1 = schemeSpecs->getErrorForDerivativeOfArbitraryScheme(1);

        map<short unsigned, map<Direction, map<vector<Position>, short>>> templatePositionsAndPointsMap = schemeBuilder.initiatePositionsAndPointsMap(maxDerivativeOrder, directions);
        schemeBuilder.templatePositionsAndPoints(1, errorOrderDerivative1, directions, templatePositionsAndPointsMap[1]);
        auto maxNeighbours = schemeBuilder.getMaximumNumberOfPointsForArbitrarySchemeType();

        for (auto &node: *nodes) {

            auto graph = IsoParametricNodeGraph(node, maxNeighbours, parametricCoordsMap, nodesPerDirection, false);
            auto availablePositionsAndDepth = graph.getColinearPositionsAndPoints(directions);
            auto nodeMetrics = make_shared<Metrics>(node, dimensions());
            //Loop through all the directions to find g_i = d(x_j)/d(x_i), g^i = d(x_i)/d(x_j)
            for (auto &directionI: directions) {
                auto i = spatialDirectionToUnsigned[directionI];
                auto covariantBaseVectorI = NumericalVector<double>(directions.size(), 0);
                auto contravariantBaseVectorI = NumericalVector<double>(directions.size(), 0);


                for (auto &directionJ : directions){
                    auto j = spatialDirectionToUnsigned[directionJ];

                    //Check if the available positions are qualified for the current derivative order
                    auto qualifiedPositions = schemeBuilder.getQualifiedFromAvailable(
                            availablePositionsAndDepth[directionJ], templatePositionsAndPointsMap[1][directionJ]);

                    auto graphFilter = map<Position, unsigned short>();
                    for (auto &tuple: qualifiedPositions) {
                        for (auto &point: tuple.first) {
                            graphFilter.insert(pair<Position, unsigned short>(point, tuple.second));
                        }
                    }
                    auto filteredNodeGraph = graph.getNodeGraph(graphFilter);

                    auto parametricCoords = (graph.*colinearNodes)(Parametric, filteredNodeGraph);
                    auto templateCoords = (graph.*colinearNodes)(coordinateSystem, filteredNodeGraph);

                    //Get the FD scheme weights for the current direction
/*                    auto covariantWeights = scheme.weights;
                    auto contravariantWeights = covariantWeights;*/
                    

                    auto taylorPointParametric = (*node->coordinates.getPositionVector(Parametric))[i];
                    auto covariantWeights = calculateWeightsOfDerivativeOrder(
                            *parametricCoords[directionJ][j]->getVectorSharedPtr(), 1, taylorPointParametric);

                    auto taylorPointTemplate = (*node->coordinates.getPositionVector(coordinateSystem))[i];
                    auto contravariantWeights = calculateWeightsOfDerivativeOrder(
                            *templateCoords[directionJ][j]->getVectorSharedPtr(), 1, taylorPointTemplate);

                    
                    //Check if the number of weights and the number of nodes match
                    if (covariantWeights.size() != parametricCoords[directionJ][j]->size()) {
                        throw std::runtime_error(
                                "Number of weights and number of template nodal coords do not match"
                                " for node " + to_string(*node->id.global) +
                                " in direction " + to_string(directionI) +
                                " Cannot calculate covariant base vectors");
                    }

                    if (contravariantWeights.size() != templateCoords[directionJ][j]->size()) {
                        throw std::runtime_error(
                                "Number of weights and number of parametric nodal coords do not match"
                                " for node " + to_string(*node->id.global) +
                                " in direction " + to_string(directionI) +
                                " Cannot calculate contravariant base vectors");
                    }

/*                    auto covariantStep = 1.0;
                    auto contravariantStep = templateCoords[directionJ][j]->averageAbsoluteDeviationFromMean();
                    contravariantStep = pow(contravariantStep, scheme.power) * scheme.denominatorCoefficient;

                    for (int weight = 0; weight < covariantWeights.size(); weight++) {
                        covariantWeights[weight] /= covariantStep;
                        contravariantWeights[weight] /= contravariantStep;
                    }*/

                    //Covariant base vectors (dr_i/dξ_i)
                    //g_1 = {dx/dξ, dy/dξ, dz/dξ}
                    //g_2 = {dx/dη, dy/dη, dz/dη} 
                    //g_3 = {dx/dζ, dy/dζ, dz/dζ}
                    covariantBaseVectorI[i] = covariantWeights.dotProduct(templateCoords[directionJ][j]);
                    //Contravariant base vectors (dξ_i/dr_i)
                    //g^1 = {dξ/dx, dξ/dy, dξ/dz}
                    //g^2 = {dη/dx, dη/dy, dη/dz}
                    //g^3 = {dζ/dx, dζ/dy, dζ/dz}
                    contravariantBaseVectorI[i] = contravariantWeights.dotProduct(parametricCoords[directionJ][j]);
                }
                nodeMetrics->covariantBaseVectors->insert(
                        pair<Direction, NumericalVector<double>>(directionI, covariantBaseVectorI));
                nodeMetrics->contravariantBaseVectors->insert(
                        pair<Direction, NumericalVector<double>>(directionI, contravariantBaseVectorI));


            }
            nodeMetrics->calculateCovariantTensor();
            nodeMetrics->calculateContravariantTensor();
            metrics->insert(pair<unsigned, shared_ptr<Metrics> >(*node->id.global, nodeMetrics));
            nodeMetrics->covariantTensor->printFullMatrix("covariant tensor node : " + to_string(*node->id.global));
            //nodeMetrics->contravariantTensor->printFullMatrix("contravariant tensor node : " + to_string(*node->id.global));
        }
    }
    

    shared_ptr<map<vector<double>, Node*>> Mesh::getCoordinatesToNodesMap(CoordinateType coordinateType) {
        auto parametricCoordToNodeMap = make_shared<map<vector<double>, Node*>>();
        for (auto& node : *totalNodesVector) {
            auto nodalParametricCoordsSTD = *node->coordinates.getPositionVector3D(coordinateType).getVectorSharedPtr();
            parametricCoordToNodeMap->insert(pair<vector<double>, Node*>(std::move(nodalParametricCoordsSTD), node));
        }
        return parametricCoordToNodeMap;
    }

    unique_ptr<map<Node *, Position>> Mesh::getBoundaryNodeToPositionMap() const {
        auto boundaryNodeToPositionMap = make_unique<map<Node *, Position>>();
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

} // Discretization