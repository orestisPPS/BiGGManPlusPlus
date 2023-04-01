//
// Created by hal9000 on 12/17/22.
//
#pragma once

#include "../Node/Node.h"
#include<vector>
#include "../../LinearAlgebra/Array.h"
#include "../../PositioningInSpace/IsoParametricCurves.h"
using namespace Discretization;
using namespace LinearAlgebra;

namespace Discretization {

     class Mesh {
     
     public:
        //Mesh(Array<Node *> *nodes, map<Direction, int> numberOfNodesPerDirection);
        Mesh();
        
        virtual ~Mesh();
                
        //map<Direction, unsigned > *numberOfNodesPerDirection;
        map<Direction, unsigned > numberOfNodesPerDirection;

        map<Position, vector<Node*>*>* boundaryNodes;
        
        vector<Node*>* internalNodes;
        
        bool isInitialized;

        //---------------Implemented parent class methods--------------
        
        unsigned totalNodes();
        
        //Returns the node pointer of the node with the given global ID
        //Ιf the node does not exist, returns nullptr
        Node* nodeFromID(unsigned ID);
        
        
        //-----------------Virtual parent class methods-----------------  

        virtual unsigned dimensions();
        
        virtual SpaceEntityType space();

        virtual Node* node(unsigned i);
    
        virtual Node* node(unsigned i, unsigned j);
    
        virtual Node* node(unsigned i, unsigned j, unsigned k);
        
        IsoParametricCurves* isoParametricCurves;
        
        virtual void printMesh();
        

        
     protected:
        Array<Node *> *_nodesMatrix;
        
        void initialize();
                  
        //Adds the boundary nodes of the  mesh to a map pointer of enum Position and vector pointers of node pointers
        virtual map<Position, vector<Node*>*> *addDBoundaryNodesToMap();
        
        //Adds the internal nodes of the mesh to a vector pointer of node pointers
        virtual vector<Node*>* addInternalNodesToVector();
        
        virtual IsoParametricCurves* createIsoParametricCurves();
        
        void categorizeNodes();
         
        void createNumberOfNodesPerDirectionMap();
        
        void cleanMeshDataStructures();
    };
}