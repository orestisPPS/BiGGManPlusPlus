//
// Created by hal9000 on 4/10/23.
//

#ifndef UNTITLED_GHOSTPSEUDOMESH_H
#define UNTITLED_GHOSTPSEUDOMESH_H

#include "../Mesh.h"

namespace Discretization {

    class GhostPseudoMesh {
        
    public:
        
        GhostPseudoMesh(Mesh* targetMesh, map<Direction, unsigned>* ghostNodesPerDirection);
        
        ~GhostPseudoMesh();
        
        Array<Node*>* ghostedNodesMatrix;
        
        list<Node*>* allNodesList;
        
        list<Node*>* ghostNodesList;

        map<Direction, unsigned>* ghostNodesPerDirection;
        
        map<vector<double>, Node*> *parametricCoordToNodeMap;
        
        Mesh* targetMesh;

        unsigned dimensions() const;

        SpaceEntityType space() const;

        virtual Node* node(unsigned i);

        virtual Node* node(unsigned i, unsigned j);

        virtual Node* node(unsigned i, unsigned j, unsigned k);
        
    protected:
        
        virtual Array<Node*>* createGhostedNodesMatrix();

        map<vector<double>, Node*> * createParametricCoordToNodeMap();
        
        void initialize();
        
    private:
        


    };

} // Discretization

#endif //UNTITLED_GHOSTPSEUDOMESH_H
