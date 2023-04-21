//
// Created by hal9000 on 3/11/23.
//

#ifndef UNTITLED_MESH3D_H
#define UNTITLED_MESH3D_H

#include "Mesh.h"

namespace Discretization {

    class Mesh3D : public Mesh {
    public:
        Mesh3D(Array<Node *> *nodes);

        ~Mesh3D();

        unsigned dimensions() override;

        SpaceEntityType space() override;

        vector<Direction> directions() override;

        Node* node(unsigned i) override;

        Node* node(unsigned i, unsigned j) override;

        Node* node(unsigned i, unsigned j, unsigned k) override;

        map<vector<double>, Node*>* createParametricCoordToNodesMap() override;
        
        void printMesh() override;
        
    protected:
        
        map<Position, vector<Node*>*> *addDBoundaryNodesToMap() override;
        
        vector<Node*>* addInternalNodesToVector() override;

        vector<Node*>* addTotalNodesToVector() override;
        
        Metrics* calculateNodeMetrics(Node* node, CoordinateType coordinateSystem) override;
        
        GhostPseudoMesh* createGhostPseudoMesh(unsigned ghostLayerDepth) override;
    };

} // Discretization

#endif //UNTITLED_MESH3D_H
