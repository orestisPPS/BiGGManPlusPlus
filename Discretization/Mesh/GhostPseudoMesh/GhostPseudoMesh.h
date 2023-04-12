//
// Created by hal9000 on 4/10/23.
//

#ifndef UNTITLED_GHOSTPSEUDOMESH_H
#define UNTITLED_GHOSTPSEUDOMESH_H

#include "../Mesh.h"

namespace Discretization {

    class GhostPseudoMesh {
        
    public:
        
        GhostPseudoMesh(Array<Node*>* ghostedNodesMatrix, list<Node*>* ghostNodesList,
                        map<Direction, unsigned>* ghostNodesPerDirection,
                        map<vector<double>, Node*>* parametricCoordToNodeMap);
        
        ~GhostPseudoMesh();
        
        Array<Node*>* ghostedNodesMatrix;
        
        list<Node*>* ghostNodesList;

        map<Direction, unsigned>* ghostNodesPerDirection;
        
        map<vector<double>, Node*> *parametricCoordToNodeMap;
        
    protected:
        
        virtual Array<Node*>* createGhostedNodesMatrix();

        map<vector<double>, Node*> * createParametricCoordToNodeMap();
        
        void updateParametricCoordToNodeMap() const;
        
        void initialize();
        
    private:
        


    };

} // Discretization

#endif //UNTITLED_GHOSTPSEUDOMESH_H
