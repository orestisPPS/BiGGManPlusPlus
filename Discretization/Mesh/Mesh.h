//
// Created by hal9000 on 12/17/22.
//
#pragma once

#include "../Node/Node.h"
#include<vector>
#include "../../LinearAlgebra/Array.h"
#include "../../StructuredMeshGeneration/MeshSpecs.h"
#include "Metrics/Metrics.h"
#include "GhostPseudoMesh/GhostPseudoMesh.h"
#include "../../LinearAlgebra/Transformations.h"

using namespace Discretization;
using namespace StructuredMeshGenerator;
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
        
        MeshSpecs* specs;
        
        map<Node*, Metrics*> *metrics;


        //---------------Implemented parent class methods--------------
        
        unsigned totalNodes();
        
        //Returns the node pointer of the node with the given global ID
        //Ιf the node does not exist, returns nullptr
        Node* nodeFromID(unsigned ID);
        
        
        //-----------------Virtual parent class methods-----------------  

         void initialize();
        
        virtual unsigned dimensions();
        
        virtual SpaceEntityType space();

        virtual Node* node(unsigned i);
    
        virtual Node* node(unsigned i, unsigned j);
    
        virtual Node* node(unsigned i, unsigned j, unsigned k);

        virtual map<vector<double>, Node*>* createParametricCoordToNodesMap();
        
        virtual void printMesh();
        
     protected:
        Array<Node *> *_nodesMatrix;

        map<unsigned, Node*>* _nodesMap;
        
        map<unsigned, Node*>* createNodesMap() const;
                  
        //Adds the boundary nodes of the  mesh to a map pointer of enum Position and vector pointers of node pointers
        virtual map<Position, vector<Node*>*> *addDBoundaryNodesToMap();
        
        //Adds the internal nodes of the mesh to a vector pointer of node pointers
        virtual vector<Node*>* addInternalNodesToVector();
        
        //Adds the total nodes of the mesh to a vector pointer of node pointers
        virtual vector<Node*>* addTotalNodesToVector();
        
        void categorizeNodes();
         
        void createNumberOfNodesPerDirectionMap();
        
        void cleanMeshDataStructures();
        
        // Calculates the metrics of all the nodes based on the given coordinate system.
        // If coordinateSystem is Template then the metrics are calculated based on the template coordinate system before
        // the final coordinate system is calculated.
        // If coordinateSystem is Natural then the metrics are calculated based on the final calculated coordinate system.
        void calculateMeshMetrics(CoordinateType coordinateSystem);

        map<Direction, unsigned>* createNumberOfGhostNodesPerDirectionMap(unsigned ghostLayerDepth);
        
        virtual Metrics* calculateNodeMetrics(Node* node, CoordinateType coordinateSystem);
        
        virtual GhostPseudoMesh* createGhostPseudoMesh(unsigned ghostLayerDepth);


        
        

    };
}