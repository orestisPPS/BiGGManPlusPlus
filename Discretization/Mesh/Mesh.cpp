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
    
    void Mesh::calculateMeshMetrics(CoordinateType coordinateSystem){

    }
    


    map<Direction, unsigned>* Mesh:: createNumberOfGhostNodesPerDirectionMap(unsigned ghostLayerDepth){
        auto numberOfGhostNodesPerDirection = new map<Direction, unsigned>();
        for (auto &direction : directions()) {
            numberOfGhostNodesPerDirection->insert(pair<Direction, unsigned>(direction, 2*ghostLayerDepth));
        }
        return numberOfGhostNodesPerDirection;

    }

    void Mesh::calculateMeshMetrics(CoordinateType coordinateSystem) {
        if (isInitialized) {
            //Initialize Mesh Metrics map
            metrics = new map<Node*, Metrics*>();

            // GhostMesh depth for metrics calculation is metricsOrder - 1,
            // since the boundary nodes need metricsOrder - 1 ghost nodes for central finite difference scheme
            unsigned depth = specs->metricsOrder - 1;
            //auto ghostNodesPerDirection = createNumberOfGhostNodesPerDirectionMap(depth);

            auto ghostMesh = createGhostPseudoMesh(specs->metricsOrder - 1);

            //Create Scheme Specs. Metrics are calculated by a central ("diamond") scheme
            //Since the ghost mesh is initialized with a depth metricsOrder - 1, all information are provided.
            map<Direction, tuple<FDSchemeType, int>> schemeTypeAndOrderAtDirection;
            for (auto &direction : directions()) {
                schemeTypeAndOrderAtDirection.insert(pair<Direction, tuple<FDSchemeType, int>>
                                                             (direction, make_tuple(FDSchemeType::Central, specs->metricsOrder)));
            }
            auto schemeSpecs = new FDSchemeSpecs(schemeTypeAndOrderAtDirection);

            //March through all the nodes of the mesh and calculate its metrics
            for (auto &node : *totalNodesVector) {
                metrics->insert(pair<Node*, Metrics*>(node, calculateNodeMetrics(node, coordinateSystem)));
                
                //Find Scheme for node
                
                

            }
        }
        else
            throw std::runtime_error("Mesh has not been initialized");
    }
    
    Metrics* Mesh::calculateNodeMetrics(Node* node, CoordinateType coordinateSystem) {
        return nullptr;
    }
    
    GhostPseudoMesh* Mesh::createGhostPseudoMesh(unsigned ghostLayerDepth) {
        return nullptr;
    }
    
} // Discretization