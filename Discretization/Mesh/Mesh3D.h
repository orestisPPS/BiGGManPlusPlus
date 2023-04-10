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

        Node* node(unsigned i) override;

        Node* node(unsigned i, unsigned j) override;

        Node* node(unsigned i, unsigned j, unsigned k) override;

        Mesh3D* createGhostMesh(map<Direction, unsigned> ghostNodesPerDirection) override;

        void printMesh() override;
        
    protected:
        
        map<Position, vector<Node*>*> *addDBoundaryNodesToMap() override;
        
        vector<Node*>* addInternalNodesToVector() override;

        vector<Node*>* addTotalNodesToVector() override;
    };

} // Discretization

#endif //UNTITLED_MESH3D_H
