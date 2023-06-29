//
// Created by hal9000 on 3/11/23.
//

#ifndef UNTITLED_MESH1D_H
#define UNTITLED_MESH1D_H

#include "Mesh.h"
#include "GhostPseudoMesh/GhostPseudoMesh.h"

namespace Discretization {

    class Mesh1D : public Mesh {
    public:
        Mesh1D(shared_ptr<Array<Node*>>nodes);
        
        ~Mesh1D();

        unsigned dimensions() override;
        
        SpaceEntityType space() override;
        
        vector<Direction> directions() override;
                
        Node* node(unsigned i) override;
        
        Node* node(unsigned i, unsigned j) override;
        
        Node* node(unsigned i, unsigned j, unsigned k) override;

        shared_ptr<map<vector<double>, Node*>> createParametricCoordToNodesMap() override;
        
        void printMesh() override;
        
        
    protected:
        
        shared_ptr<map<Position, shared_ptr<vector<Node*>>>>_addDBoundaryNodesToMap() override;

        unique_ptr<vector<Node*>> getInternalNodesVector() override;
        
        shared_ptr<vector<Node*>> _addTotalNodesToVector() override;
        
        GhostPseudoMesh* _createGhostPseudoMesh(unsigned ghostLayerDepth) override;
        
    };

} // Discretization

#endif //UNTITLED_MESH1D_H
