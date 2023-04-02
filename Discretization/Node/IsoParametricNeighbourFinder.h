//
// Created by hal9000 on 4/2/23.
//

#ifndef UNTITLED_ISOPARAMETRICNEIGHBOURFINDER_H
#define UNTITLED_ISOPARAMETRICNEIGHBOURFINDER_H

#include "../Mesh/Mesh.h"
#include "../../Utility/Calculators.h"
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
        
        map<Node*, map<Position, vector<Node*>>> getNeighbourNodes(unsigned nodeId);
        
        
        Node* getNeighbourAtPosition(unsigned nodeId, Position position);
        

        map<Position, vector<Node*>> getNeighbourNodes(unsigned nodeId, unsigned  depth);
        
        map<Position, vector<Node*>> getCustomDepthNeighbourNodes(unsigned nodeId, map<Position, unsigned> depthMap);
        
        map<Position, vector<vector<DegreeOfFreedom*>>> getAllNeighbourDOF(unsigned nodeId, unsigned depth);

        map<Position, vector<DegreeOfFreedom*>> getSpecificNeighbourDOF(unsigned nodeId, DOFType dofType, unsigned depth);
        
        map<Position, vector<DegreeOfFreedom*>> getSpecificNeighbourDOF(unsigned nodeId, DOFType dofType,
                                                                        ConstraintType constraintType, unsigned depth);
        
    private:
        map<vector<double>, Node*> _parametricCoordinatesToNodeMap;
        
        Mesh* _mesh;
        
        
        // Maps all the parametric coordinates of mesh to the node pointer they belong to.
        map<vector<double>, Node*> _createParametricCoordinatesToNodeMap();
        
        void _addNodeToNeighboursIfParametricCoordsExists(Position position, vector<double>& parametricCoords, map<Position,
                                                          vector<Node*>> &neighbourNodesMap);
        
    };

} // Discretization

#endif //UNTITLED_ISOPARAMETRICNEIGHBOURFINDER_H
