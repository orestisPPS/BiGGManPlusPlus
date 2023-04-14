//
// Created by hal9000 on 3/11/23.
//

#ifndef UNTITLED_MESH2D_H
#define UNTITLED_MESH2D_H

#include "Mesh.h"

namespace Discretization {

    class Mesh2D : public Mesh {
        
    public:
        Mesh2D(Array<Node *> *nodes);
        
        ~Mesh2D();
        
        unsigned dimensions() override;
        
        SpaceEntityType space() override;
        
        Node* node(unsigned i) override;
        
        Node* node(unsigned i, unsigned j) override;
        
        Node* node(unsigned i, unsigned j, unsigned k) override;

        map<vector<double>, Node*>* createParametricCoordToNodesMap() override;
        
        void printMesh() override;
        
    protected:

        map<Position, vector<Node*>*> *addDBoundaryNodesToMap() override;

        vector<Node*>* addInternalNodesToVector() override;

        vector<Node*>* addTotalNodesToVector() override;
        
        void calculateMeshMetrics(CoordinateType coordinateSystem) override;
        
        Metrics* calculateNodeMetrics(Node* node, CoordinateType coordinateSystem) override;
        
        GhostPseudoMesh* createGhostPseudoMesh(unsigned ghostLayerDepth) override;

    };

} // Discretization

#endif //UNTITLED_MESH2D_H
