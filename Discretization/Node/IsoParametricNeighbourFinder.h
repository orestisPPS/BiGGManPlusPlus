//
// Created by hal9000 on 4/2/23.
//

#ifndef UNTITLED_ISOPARAMETRICNEIGHBOURFINDER_H
#define UNTITLED_ISOPARAMETRICNEIGHBOURFINDER_H

#include "../Mesh/Mesh.h"
#include "../../Utility/Calculators.h"
#include "IsoparametricNodeHoodGraph.h"
#include <cmath>


namespace Discretization {

    class IsoParametricNeighbourFinder {
        
    public:
        // Static Utility class that finds the neighbours of a node in structured mesh by exploiting iso-parametric
        // curves of the mesh. Each node belongs in an iso-parametric curve where every
        // node shares a common parametric coordinate in one  or two directions. (iso-line, iso-surface)
        explicit IsoParametricNeighbourFinder(Mesh* mesh);
        
        ~IsoParametricNeighbourFinder();
        
        double maxMagnitude;
        
        
        Node* getNeighbourAtPosition(unsigned nodeId, Position position);
        
        // Returns a map pointer of the input node neighbours graph at the input depth.
        // Key : Position Enum representing the position of the neighbour node relative to the input node.
        // Value : Map of the neighbour nodes at all the input depth.
        //       Key : Depth of the neighbour node.
        //       Value : Neighbour node pointer.
        // Warning : Always deallocate the returned pointer and the value map pointers.
        map<Position, map<unsigned, Node*>>* getAllNeighbourNodes(unsigned nodeId, unsigned  depth);
        
        //map<Position, vector<Node*>> getCustomDepthNeighbourNodes(unsigned nodeId, map<Position, unsigned> depthMap);

        // Returns a map pointer of the input node neighbours Degrees Of Freedom graph at the input depth.
        // Key : Position Enum representing the position of the neighbour node relative to the input node.
        // Value : Map of the neighbour nodes at all the input depth.
        //       Key : Depth of the neighbour node.
        //       Value : Neighbour node DegreesOfFreedom Vector Pointer.
        // Warning : Always deallocate the returned pointer and the value map pointers.
        map<Position, map<unsigned, vector<DegreeOfFreedom*>>*>* getAllNeighbourDOF(unsigned nodeId, unsigned depth);

        // Returns a map pointer of the input node neighbours Degrees Of Freedom graph at the input depth.
        // Key : Position Enum representing the position of the neighbour node relative to the input node.
        // Value : Map of the neighbour nodes at all the input depth.
        //       Key : Depth of the neighbour node.
        //       Value : Specific DOFType Neighbour node DegreesOfFreedom Pointer.
        // Warning : Always deallocate the returned pointer and the value map pointers.
        map<Position, map<unsigned, DegreeOfFreedom*>>* getSpecificNeighbourDOF(unsigned nodeId, DOFType dofType,
                                                                                unsigned depth);
    
        // Returns a map pointer of a specific Degree Of Freedom of the input node neighbours at the input depth.
        // Key : Position Enum representing the position of the neighbour node relative to the input node.
        // Value : Map of the neighbour nodes at all the input depth.
        //       Key : Depth of the neighbour node.
        //       Value : Specific DOFType and ConstraintType Neighbour node DegreesOfFreedom Pointer.
        // Warning : Always deallocate the returned pointer and the value map pointers.
        map<Position, map<unsigned, DegreeOfFreedom*>>* getSpecificNeighbourDOF(unsigned nodeId, DOFType dofType,
                                                                        ConstraintType constraintType, unsigned depth);
        
    private:
        map<vector<double>, Node*> *_parametricCoordinatesToNodeMap;
        
        Mesh* _mesh;
        
        
        // Maps all the parametric coordinates of mesh to the node pointer they belong to.
        map<vector<double>, Node*>* _createParametricCoordinatesToNodeMap();
        
        void _addNeighbourNodeIfParametricCoordsExist(Position position, vector<double>& parametricCoords,
                                                             map<Position, map<unsigned, Node*>>* neighbourNodesMap, unsigned depthSize);
        
    };

} // Discretization

#endif //UNTITLED_ISOPARAMETRICNEIGHBOURFINDER_H
