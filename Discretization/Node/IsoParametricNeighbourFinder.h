//
// Created by hal9000 on 4/2/23.
//

#ifndef UNTITLED_ISOPARAMETRICNEIGHBOURFINDER_H
#define UNTITLED_ISOPARAMETRICNEIGHBOURFINDER_H

#include "../Mesh/Mesh.h"

namespace Discretization {

    class IsoParametricNeighbourFinder {
        
    public:
        // Static Utility class that finds the neighbours of a node in structured mesh by exploiting iso-parametric
        // curves of the mesh. Each node belongs in an iso-parametric curve where every
        // node shares a common parametric coordinate in one  or two directions. (iso-line, iso-surface)
        explicit IsoParametricNeighbourFinder(Mesh* mesh);
        
        vector<double> maximumParametricCoords;
        
        map<Position, vector<double>> allPossibleNeighbourParametricCoords;
        
        Node* getNeighbourAtPosition(unsigned nodeId, Position position);
        



        map<Position, map<int, Node*>> getNeighbourNodes(unsigned nodeId, unsigned  depth);
        
        map<Position, map<int, Node*>> getCustomDepthNeighbourNodes(unsigned nodeId, map<Position, unsigned> depthMap);
        
        map<Position, map<int, vector<DegreeOfFreedom*>>> getAllNeighbourDOF(unsigned nodeId, unsigned depth);

        map<Position, map<int, DegreeOfFreedom*>> getSpecificNeighbourDOF(unsigned* nodeId, DOFType dofType, unsigned depth);
        
    private:
        map<vector<double>, Node*> _parametricCoordinatesToNodeMap;
        
        Mesh* _mesh;

        // Returns the map of the parametric coordinate vectors of all possible neighbour nodes.
        static map<Position, vector<double>>  _getAllPossibleNeighbourParametricCoords();
        
        // Maps all the parametric coordinates of the nodes in the mesh to the node pointer they belong to.
        map<vector<double>, Node*> _createParametricCoordinatesToNodeMap();
        
        
        
    };

} // Discretization

#endif //UNTITLED_ISOPARAMETRICNEIGHBOURFINDER_H
