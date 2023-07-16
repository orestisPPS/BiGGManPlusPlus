//
// Created by hal9000 on 3/11/23.
//

#include "Mesh3D.h"

#include <utility>

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
                {Back, {One, Three}},
                {Front, {Three, One}}
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
    

} // Discretization