//
// Created by hal9000 on 3/11/23.
//

#ifndef UNTITLED_MESH2D_H
#define UNTITLED_MESH2D_H

#include "Mesh.h"

namespace Discretization {

    class Mesh2D : public Mesh {
        
    public:
        Mesh2D(shared_ptr<Array<Node*>>nodes);
        
        ~Mesh2D();
        
        unsigned dimensions() override;
        
        unsigned numberOfInternalNodes() override;
        
        SpaceEntityType space() override;

        vector<Direction> directions() override;
        
        Node* node(unsigned i) override;
        
        Node* node(unsigned i, unsigned j) override;
        
        Node* node(unsigned i, unsigned j, unsigned k) override;

        shared_ptr<map<vector<double>, Node*>> createParametricCoordToNodesMap() override;

        unique_ptr<vector<Node*>> getInternalNodesVector() override;
        
        void printMesh() override;
        
        vector<double> getNormalUnitVectorOfBoundaryNode(Position boundaryPosition, Node *node) override;
        
        void createElements(ElementType elementType, unsigned int nodesPerEdge) override;
        
    protected:

        shared_ptr<map<Position, shared_ptr<vector<Node*>>>> _addDBoundaryNodesToMap() override;
        
        shared_ptr<vector<Node*>> _addTotalNodesToVector() override;
        
        GhostPseudoMesh* _createGhostPseudoMesh(unsigned ghostLayerDepth) override;

    };

} // Discretization

#endif //UNTITLED_MESH2D_H
