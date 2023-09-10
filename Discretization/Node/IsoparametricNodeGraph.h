//
// Created by hal9000 on 4/3/23.
//

#ifndef UNTITLED_ISOPARAMETRICNODEGRAPH_H
#define UNTITLED_ISOPARAMETRICNODEGRAPH_H

#include "Node.h"
#include <utility>
#include <algorithm>
#include <memory>

using namespace Discretization;

namespace Discretization {

    class IsoParametricNodeGraph {
        
    public:
        
        IsoParametricNodeGraph(Node* node, unsigned graphDepth, shared_ptr<map<NumericalVector<double>, Node*>>nodeMap,
                               map<Direction, unsigned>& nodesPerDirection, bool includeDiagonalNeighbours = true);
        
        // Returns a map ptr of the input node neighbours graph
        // Key : Position Enum representing the position of the neighbour node relative to the input node.
        // Value : Vector Node ptr of the neighbour nodes at all the input depth levels. (length = depth)
        // Warning : Always deallocate the returned pointer.
        const map<Position, vector<Node*>>& getNodeGraph() const ;

        map<Position, vector<Node*>> getNodeGraph(const map<Position,short unsigned>& customDepth) const ;
        

        // Map ptr graph with a specific Degree Of Freedom of the input node neighbours (free and constrained).
        // Key : Position Enum representing the position of the neighbour node relative to the input node.
        // Value : Vector of DegreeOfFreedom Pointers vector pointer at all the input depth levels.
        // Warning : Always deallocate the returned pointer.
        map<Position, vector<DegreeOfFreedom*>> getSpecificDOFGraph(DOFType dofType) const ;
        
        
        static map<Position, vector<DegreeOfFreedom*>>
        getSpecificDOFGraph(DOFType dofType, map<Position, vector<Node*>>& customNodeGraph) ;

        // Map ptr graph with a specific Degree Of Freedom of the input node neighbours (free or constrained).
        // Key : Position Enum representing the position of the neighbour node relative to the input node.
        // Value : Vector of DegreeOfFreedom Pointers vector pointer at all the input depth levels.
        // Warning : Always deallocate the returned pointer.
        map<Position, vector<DegreeOfFreedom*>>
        getSpecificDOFGraph(DOFType dofType, ConstraintType constraint) const ;
        
        static map<Position, vector<DegreeOfFreedom*>>
        getSpecificDOFGraph(DOFType dofType, ConstraintType constraint, map<Position, vector<Node*>>& customNodeGraph) ;
        
        // Returns a map ptr graph with all the Nodes that belong at the same parametric coordinate line as the input node.
        // Key : Direction Enum representing the direction of the parametric coordinate line.
        // Value : Vector of Node ptr at all depth nodes. They are ordered with increasing parametric coordinate value.
        //         The input node is included in the vector.
        map<Direction, vector<Node*>> getColinearNodes() const;
        
        map<Direction, vector<Node*>> getColinearNodes(map<Position, vector<Node*>>& customNodeGraph) const;
        
        map<Direction, vector<Node*>> getColinearNodesOnBoundary(map<Position, vector<Node*>>& customNodeGraph) const;
        
        //THIS MIGHT BE SERIOUSLY FUCKED UP
        vector<Node*>  getColinearNodes(Direction direction, map<Position, vector<Node*>>& customNodeGraph) const;

        map<Direction, map<vector<Position>, short unsigned>> getColinearPositionsAndPoints(vector<Direction>& availableDirections) ;
        
        map<Direction, map<vector<Position>, short unsigned>>
        getColinearPositionsAndPoints(vector<Direction>& availableDirections, map<Position, vector<Node*>>& customNodeGraph) ;

        map<Direction, vector<NumericalVector<double>>> getSameColinearNodalCoordinates(CoordinateType coordinateType) const;

        map<Direction, vector<NumericalVector<double>>>
        getSameColinearNodalCoordinates(CoordinateType coordinateType, map<Position, vector<Node*>>& customNodeGraph) const;
        
        map<Direction, vector<NumericalVector<double>>> getSameColinearNodalCoordinatesOnBoundary(CoordinateType coordinateType, map<Position, vector<Node*>>& customNodeGraph) const;
        
        map<Direction, vector<DegreeOfFreedom*>> getColinearDOF(DOFType dofType) const;
        
        map<Direction, vector<DegreeOfFreedom*>> getColinearDOF(DOFType dofType, map<Position, vector<Node*>>& customNodeGraph) const;
        
        vector<DegreeOfFreedom*> getColinearDOF(DOFType dofType, Direction direction, map<Position, vector<Node *>> &customNodeGraph) const;
        
        vector<DegreeOfFreedom*> getColinearDOFOnBoundary(DOFType dofType, Direction direction, map<Position, vector<Node *>> &customNodeGraph) const;

        map<Direction, vector<DegreeOfFreedom*>> getColinearDOFOnBoundary(DOFType dofType, map<Position, vector<Node*>>& customNodeGraph) const;



        map<Position, vector<Node*>> nodeGraph;
                
    private:

        Node* _node;
        
        unsigned int _graphDepth;
        
        shared_ptr<map<NumericalVector<double>, Node*>> _nodeMap;
        
        map<Direction, unsigned> _nodesPerDirection;
                
        void _findINeighborhoodRecursively(bool includeDiagonalNeighbours);
        
        void _findIDepthNeighborhood(unsigned int depth, NumericalVector<double>& nodeCoords);
        
        void _findIDepthNeighborhoodOnlyDiagonals(unsigned int depth, NumericalVector<double>& nodeCoords);

        void _addNeighbourNodeIfParametricCoordsExist(Position position, NumericalVector<double> &parametricCoords);
        
        static vector<Node*> _mergeAndSortColinearNodes(vector<Node*>& nodesDirection1, vector<Node*>& nodesDirection2, Node* node);
        
        static vector<Node*> _mergeAndSortColinearNodes(vector<Node*>& nodesDirection1, Node* node);
    };

} // Node

#endif //UNTITLED_ISOPARAMETRICNODEGRAPH_H
