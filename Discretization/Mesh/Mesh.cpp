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
            numberOfGhostNodesPerDirection->insert(pair<Direction, unsigned>(direction, 2*ghostLayerDepth));
        }
        return numberOfGhostNodesPerDirection;

    }

    void Mesh::calculateMeshMetrics(CoordinateType coordinateSystem) {
        if (isInitialized) {
            //Initialize Mesh Metrics map
            metrics = new map<Node*, Metrics*>();
            
            //Create Scheme Specs. Metrics are calculated by a central ("diamond") scheme
            auto schemeSpecs = new FDSchemeSpecs(Central, specs->metricsOrder - 1, directions());
            //Create Scheme Builder to gain access to utility functions for the scheme creation
            auto schemeBuilder = new FiniteDifferenceSchemeBuilder(schemeSpecs);
            // Initiate GhostPseudoMesh
            auto ghostMesh = createGhostPseudoMesh(schemeBuilder->getNumberOfGhostNodesNeeded());
            
            //March through all the nodes of the mesh and calculate its metrics
            for (auto &node : *totalNodesVector) {
                //Find Neighbors in a manner that applies the same numerical scheme to all nodes
                auto neighbors = schemeBuilder->getNumberOfDiagonalNeighboursNeeded();
                //Initiate Node Graph
                auto graph = new IsoParametricNodeGraph(node, schemeBuilder->getNumberOfGhostNodesNeeded(),
                                                ghostMesh->parametricCoordToNodeMap, numberOfNodesPerDirection);
                
                //Get the adjusted node graph that contains only the nodes that are needed to calculate the FD scheme
                //provided by the scheme builder
                auto nodeGraph = graph->getNodeGraph(neighbors);
                
                //Create finite difference scheme
                
                
                //Deallocate memory
                delete nodeGraph;
                nodeGraph = nullptr;
                delete graph;
                graph = nullptr;
                
            }
            delete schemeBuilder;
            schemeBuilder = nullptr;
            delete schemeSpecs;
            schemeSpecs = nullptr;
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