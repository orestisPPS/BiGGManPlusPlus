//
// Created by hal9000 on 3/11/23.
//

#ifndef UNTITLED_MESH3D_H
#define UNTITLED_MESH3D_H

#include "Mesh.h"

namespace Discretization {

    class Mesh3D : public Mesh {
    public:
        Mesh3D(shared_ptr<Array<Node*>>nodes);

        ~Mesh3D();

        unsigned dimensions() override;

        SpaceEntityType space() override;

        vector<Direction> directions() override;

        Node* node(unsigned i) override;

        Node* node(unsigned i, unsigned j) override;

        Node* node(unsigned i, unsigned j, unsigned k) override;

        shared_ptr<map<vector<double>, Node*>> createParametricCoordToNodesMap() override;
        
        void printMesh() override;
        
    protected:
        
        shared_ptr<map<Position, shared_ptr<vector<Node*>>>>addDBoundaryNodesToMap() override;
        
        shared_ptr<vector<Node*>> addInternalNodesToVector() override;

        shared_ptr<vector<Node*>> addTotalNodesToVector() override;
        
        GhostPseudoMesh* createGhostPseudoMesh(unsigned ghostLayerDepth) override;
    };

} // Discretization

#endif //UNTITLED_MESH3D_H
