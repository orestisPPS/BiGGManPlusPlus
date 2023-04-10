//
// Created by hal9000 on 12/17/22.
//
#pragma once

#include "../Node/Node.h"
#include<vector>
#include "../../LinearAlgebra/Array.h"
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
        
        vector<Node*>* internalNodesVector;
        
        vector<Node*>* totalNodesVector;
                
        bool isInitialized;


        //---------------Implemented parent class methods--------------
        
        unsigned totalNodes();
        
        //Returns the node pointer of the node with the given global ID
        //Ιf the node does not exist, returns nullptr
        Node* nodeFromID(unsigned ID);
        
        
        //-----------------Virtual parent class methods-----------------  

         void initialize(bool categorizeNodes);
        
        virtual unsigned dimensions();
        
        virtual SpaceEntityType space();

        virtual Node* node(unsigned i);
    
        virtual Node* node(unsigned i, unsigned j);
    
        virtual Node* node(unsigned i, unsigned j, unsigned k);

         // Creates a mesh that contains ghost nodes beyond the boundary of the mesh for the purpose of
         // more accurate calculation of the boundary conditions and easier calculation of the mesh metrics.
         virtual Mesh* createGhostMesh(map<Direction, unsigned> ghostNodesPerDirection);

        virtual void printMesh();
        
     protected:
        Array<Node *> *_nodesMatrix;
        
        Array<Node *> *_ghostedNodesMatrix;
        
        map<unsigned, Node*>* _nodesMap;
        
        map<unsigned, Node*>* _ghostedNodesMap;
        
        map<unsigned, Node*>* createNodesMap() const;
        
        //Creates an array ptr of node ptr that contains both the ghost and the regular nodes of the mesh
        virtual Mesh* createGhostedSelf();
                  
        //Adds the boundary nodes of the  mesh to a map pointer of enum Position and vector pointers of node pointers
        virtual map<Position, vector<Node*>*> *addDBoundaryNodesToMap();
        
        //Adds the internal nodes of the mesh to a vector pointer of node pointers
        virtual vector<Node*>* addInternalNodesToVector();
        
        //Adds the total nodes of the mesh to a vector pointer of node pointers
        virtual vector<Node*>* addTotalNodesToVector();
        
        void categorizeNodes();
         
        void createNumberOfNodesPerDirectionMap();
        
        void cleanMeshDataStructures();
    };
}