//
// Created by hal9000 on 12/17/22.
//

#include "Mesh.h"
#include "../../LinearAlgebra/FiniteDifferences/FDSchemeType.h"
#include "../../LinearAlgebra/FiniteDifferences/FDSchemeSpecs.h"

using namespace  Discretization;

namespace Discretization {
    
    Mesh::Mesh() {
        isInitialized = false;
        _nodesMatrix = nullptr;
        boundaryNodes = nullptr;
        internalNodesVector = nullptr;
        totalNodesVector = nullptr;
        _nodesMap = nullptr;
        
    }
        
    Mesh::~Mesh() {
        delete _nodesMatrix;
        _nodesMatrix = nullptr;
        delete boundaryNodes;
        boundaryNodes = nullptr;
        delete internalNodesVector;
        internalNodesVector = nullptr;
        delete totalNodesVector;
        totalNodesVector = nullptr;
    }
        
    unsigned Mesh::totalNodes() {
        if (isInitialized)
            return _nodesMatrix->size();
        else
            throw std::runtime_error("Mesh has not been initialized");
    }
    
    Node* Mesh::nodeFromID(unsigned ID) {
        if (isInitialized)
            return _nodesMap->at(ID);
        else
            return nullptr;
    }
        
    unsigned Mesh::dimensions(){ return 0;}
    
    SpaceEntityType Mesh::space() {
        return NullSpace;
    }
    
    vector<Direction> Mesh::directions() {
        return {};
    }
    
    Node* Mesh::node(unsigned i) {
        return nullptr;
     }
    
    Node* Mesh::node(unsigned i, unsigned j) {
        return nullptr;
    }
    
    Node* Mesh::node(unsigned i, unsigned j, unsigned k) {
        return nullptr;
    }
    
    map<vector<double>, Node*>* Mesh::createParametricCoordToNodesMap() {
        return nullptr;
    }

    void Mesh::printMesh() { }

    void Mesh::initialize() {
        isInitialized = true;
        createNumberOfNodesPerDirectionMap();
        categorizeNodes();
    }
    
    map<Position, vector<Node*>*>* Mesh::addDBoundaryNodesToMap() {
        return nullptr;
    }
    
    vector<Node*>* Mesh::addInternalNodesToVector() {
        return nullptr;
    }
    
    vector<Node*>* Mesh::addTotalNodesToVector() {
        return nullptr;        
    }
    
    
    void createNumberOfNodesPerDirectionMap() { }
    


    
    void Mesh::categorizeNodes() {
        if (isInitialized) {
            boundaryNodes = addDBoundaryNodesToMap();
            internalNodesVector = addInternalNodesToVector();
            totalNodesVector = addTotalNodesToVector();
        }
        else
            throw std::runtime_error("Mesh has not been initialized");
    }
    
    void Mesh::createNumberOfNodesPerDirectionMap() {
        if (isInitialized) {
            numberOfNodesPerDirection = map<Direction, unsigned>();
            numberOfNodesPerDirection[One] = _nodesMatrix->numberOfColumns();
            numberOfNodesPerDirection[Two] = _nodesMatrix->numberOfRows();
            numberOfNodesPerDirection[Three] = _nodesMatrix->numberOfAisles();
        }
        else
            throw std::runtime_error("Mesh has not been initialized");
    }
    
    map<unsigned , Node*>* Mesh::createNodesMap() const {
        if (isInitialized) {
            auto nodesMap = new map<unsigned , Node*>();
            for (auto &node : *totalNodesVector) {
                
                nodesMap->insert(pair<unsigned, Node*>(*node->id.global, node));
            }
            return nodesMap;
        }
        else
            throw std::runtime_error("Mesh has not been initialized");
    }
    
    void Mesh::cleanMeshDataStructures() {
        //search all boundaryNodes map and deallocate all vector pointer values
        for (auto &boundary : *boundaryNodes) {
            delete boundary.second;
            boundary.second = nullptr;
        }
        delete boundaryNodes;
        boundaryNodes = nullptr;

        //Deallocate internalNodesVector vector
        delete internalNodesVector;
        internalNodesVector = nullptr;
    }

    map<Direction, unsigned>* Mesh:: createNumberOfGhostNodesPerDirectionMap(unsigned ghostLayerDepth){
        auto numberOfGhostNodesPerDirection = new map<Direction, unsigned>();
        for (auto &direction : directions()) {
            numberOfGhostNodesPerDirection->insert(pair<Direction, unsigned>(direction, ghostLayerDepth));
        }
        return numberOfGhostNodesPerDirection;

    }

    void Mesh::calculateMeshMetrics(CoordinateType coordinateSystem, bool isUniformMesh) {
        if (isInitialized) {
            //Initialize Mesh Metrics map
            metrics = new map<Node*, Metrics*>();
            
            //Create Scheme Specs. Metrics are calculated by a central ("diamond") scheme
            auto schemeSpecs = new FDSchemeSpecs(Central, specs->metricsOrder, directions());
            //Create Scheme Builder to gain access to utility functions for the scheme creation
            auto schemeBuilder = new FiniteDifferenceSchemeBuilder(schemeSpecs);
            // Initiate GhostPseudoMesh
            auto ghostMesh = createGhostPseudoMesh(schemeBuilder->getNumberOfGhostNodesNeeded());
            
            //March through all the nodes of the mesh and calculate its metrics
            for (auto &node : *totalNodesVector) {
                cout<<"Node : "<<*node->id.global<<endl;
                //Find Neighbors in a manner that applies the same numerical scheme to all nodes
                auto neighbours = schemeBuilder->getNumberOfDiagonalNeighboursNeeded();
                //Initiate Node Graph
                auto graph = new IsoParametricNodeGraph(node, schemeBuilder->getNumberOfGhostNodesNeeded(),
                                                ghostMesh->parametricCoordToNodeMap, numberOfNodesPerDirection, false);
                
                //Get the adjusted node graph that contains only the nodes that are needed to calculate the FD scheme
                //provided by the scheme builder
                auto nodeGraph = graph->getNodeGraph(neighbours);

                //Get the co-linear nodal coordinates (Left-Right -> 1, Up-Down -> 2, Front-Back -> 3)
                auto templateCoordsMap= graph->getColinearNodalCoordinate(coordinateSystem);
                auto parametricCoordsMap = graph->getColinearNodalCoordinate(Parametric);
                //Initialize Metrics class for the current node.
                auto nodeMetrics = new Metrics(node, dimensions());
                
                auto directionsVector = directions();
                //March through all the directions I (g_i = d(x_j)/d(x_i))
                for (auto &directionI : directionsVector) {
                    
                    //Initialize the weights vector. Their values depend on whether the mesh is uniform or not.
                    //For uniform meshes, the weights are the same for all nodes in all directions and for non-uniform
                    //meshes, the weights are different for each node in each direction. (Linear System Solution needed)
                    auto covariantWeights = vector<double>();
                    auto contravariantWeights = vector<double>();
                    
                    if (isUniformMesh){
                        //Get the FD scheme weights for the current direction
                        covariantWeights = schemeBuilder->getSchemeWeightsAtDirection(directionI);
                        contravariantWeights = covariantWeights;
                        //Check if the number of weights and the number of nodes match
                        if (covariantWeights.size() != templateCoordsMap[directionI].size()){
                            throw std::runtime_error("Number of weights and number of template nodal coords do not match"
                                                     " for node " + to_string(*node->id.global) +
                                                     " in direction " + to_string(directionI) +
                                                     "Cannot calculate covariant base vectors");
                        }

                        if (covariantWeights.size() != parametricCoordsMap[directionI].size()){
                            throw std::runtime_error("Number of weights and number of parametric nodal coords do not match"
                                                     " for node " + to_string(*node->id.global) +
                                                     " in direction " + to_string(directionI) +
                                                     "Cannot calculate contravariant base vectors");
                        }
                    }
                    else{
                        covariantWeights = FiniteDifferenceSchemeWeightsCalculator::
                        calculateWeights(1, templateCoordsMap[directionI]);
                        contravariantWeights = FiniteDifferenceSchemeWeightsCalculator::
                        calculateWeights(1, parametricCoordsMap[directionI]);
                    }
                    for (auto &directionJ : directionsVector){
                        //Covariant base vectors (dr_i/dξ_i)
                        //g_1 = {dx/dξ, dy/dξ, dz/dξ}
                        //g_2 = {dx/dη, dy/dη, dz/dη}
                        //g_3 = {dx/dζ, dy/dζ, dz/dζ}
                        nodeMetrics->covariantBaseVectors->at(directionJ)[spatialDirectionToUnsigned[directionJ]] =
                                VectorOperations::dotProduct(covariantWeights, templateCoordsMap[directionJ]);
                        //Contravariant base vectors (dξ_i/dr_i)
                        //g^1 = {dξ/dx, dξ/dy, dξ/dz}
                        //g^2 = {dη/dx, dη/dy, dη/dz}
                        //g^3 = {dζ/dx, dζ/dy, dζ/dz}
                        nodeMetrics->contravariantBaseVectors->at(directionJ)[spatialDirectionToUnsigned[directionJ]] =
                                VectorOperations::dotProduct(contravariantWeights, parametricCoordsMap[directionJ]);
                    }
                }
                nodeMetrics->calculateCovariantTensor();
                nodeMetrics->calculateContravariantTensor();
                metrics->insert(pair<Node*, Metrics*>(node, nodeMetrics));
                
                //Deallocate memory
                delete nodeGraph;
                nodeGraph = nullptr;
                delete graph;
                graph = nullptr;
            }
            delete ghostMesh;
            delete schemeBuilder;
            schemeBuilder = nullptr;
            delete schemeSpecs;
            schemeSpecs = nullptr;
            
            for (auto &node : *metrics) {
                //node.first->printNode();
                cout<<"Node "<<*node.first->id.global<<endl;
                cout<<"contravariant tensor"<<endl;
                node.second->covariantTensor->print();
                cout<<"contravariant tensor"<<endl;
                node.second->contravariantTensor->print();
                cout<<"---------------------------------"<<endl;

            }
            
            
        }
        else
            throw std::runtime_error("Mesh has not been initialized");
    }

    GhostPseudoMesh* Mesh::createGhostPseudoMesh(unsigned ghostLayerDepth) {
        return nullptr;
    }
    
} // Discretization