//
// Created by hal9000 on 4/3/23.
//

#ifndef UNTITLED_ISOPARAMETRICNODEGRAPH_H
#define UNTITLED_ISOPARAMETRICNODEGRAPH_H

#include "Node.h"
#include "../Mesh/Mesh.h"

using namespace Discretization;

namespace Discretization {

    class IsoParametricNodeGraph {
        
    public:
        
        IsoParametricNodeGraph(Node* node, unsigned graphDepth, Mesh* mesh, map<vector<double>, Node*> *nodeMap);

        // Returns a map ptr of the input node neighbours graph
        // Key : Position Enum representing the position of the neighbour node relative to the input node.
        // Value : Vector Node ptr of the neighbour nodes at all the input depth levels. (length = depth)
        // Warning : Always deallocate the returned pointer.
        map<Position, vector<Node*>>* getNodeGraph() const;

        // Returns a map ptr graph with all the Degrees Of Freedom of the input node neighbours.
        // Key : Position Enum representing the position of the neighbour node relative to the input node.
        // Value : Vector of DegreeOfFreedom Pointers vector pointer at all the input depth levels.
        // Warning : Always deallocate the returned pointer.
        map<Position, vector<vector<DegreeOfFreedom*>*>>* getAllDOFGraph() const;

        // Map ptr graph with a specific Degree Of Freedom of the input node neighbours (free and constrained).
        // Key : Position Enum representing the position of the neighbour node relative to the input node.
        // Value : Vector of DegreeOfFreedom Pointers vector pointer at all the input depth levels.
        // Warning : Always deallocate the returned pointer.
        map<Position, vector<DegreeOfFreedom*>>* getSpecificDOFGraph(DOFType dofType) const;

        // Map ptr graph with a specific Degree Of Freedom of the input node neighbours (free or constrained).
        // Key : Position Enum representing the position of the neighbour node relative to the input node.
        // Value : Vector of DegreeOfFreedom Pointers vector pointer at all the input depth levels.
        // Warning : Always deallocate the returned pointer.
        map<Position, vector<DegreeOfFreedom*>>* getSpecificDOFGraph(DOFType dofType, ConstraintType constraint) const;

        map<Position, vector<Node*>>* nodeGraph;
                
    private:

        Node* _node;
        
        unsigned int _graphDepth;
        
        Mesh* _mesh;

        map<vector<double>, Node*>* _nodeMap;
                
        void _findINeighborhoodRecursively();
        
        void _findIDepthNeighborhood(unsigned int depth, vector<double>& nodeCoords);

        void _addNeighbourNodeIfParametricCoordsExist(Position position, vector<double>& parametricCoords,  unsigned depthSize);
    };

} // Node

#endif //UNTITLED_ISOPARAMETRICNODEGRAPH_H
