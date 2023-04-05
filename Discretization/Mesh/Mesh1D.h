//
// Created by hal9000 on 3/11/23.
//

#ifndef UNTITLED_MESH1D_H
#define UNTITLED_MESH1D_H

#include "Mesh.h"

namespace Discretization {

    class Mesh1D : public Mesh {
    public:
        Mesh1D(Array<Node *> *nodes);
        
        ~Mesh1D();

        unsigned dimensions() override;
        
        SpaceEntityType space() override;
                
        Node* node(unsigned i) override;
        
        Node* node(unsigned i, unsigned j) override;
        
        Node* node(unsigned i, unsigned j, unsigned k) override;
        
        void printMesh() override;
        
    protected:
        
        map<Position, vector<Node*>*> *addDBoundaryNodesToMap() override;
        
        vector<Node*>* addInternalNodesToVector() override;
        
        vector<Node*>* addTotalNodesToVector() override;
        

        
    };

} // Discretization

#endif //UNTITLED_MESH1D_H
