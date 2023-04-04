//
// Created by hal9000 on 4/2/23.
//

#ifndef UNTITLED_ISOPARAMETRICNEIGHBOURFINDER_H
#define UNTITLED_ISOPARAMETRICNEIGHBOURFINDER_H

#include "../Mesh/Mesh.h"
#include "../../Utility/Calculators.h"
#include "IsoparametricNodeGraph.h"
#include <cmath>


namespace Discretization {

    class IsoParametricNeighbourFinder {
        
    public:
        // Utility class that finds the neighbours of a node in structured mesh by exploiting iso-parametric
        // curves of the mesh. Each node belongs in (1-3) iso-parametric curves and its neighbours belong in
        // one of the iso-parametric curves of the node. One can traverse the computational domain by moving
        // in the parametric domain like a king in chess, one step at a time.
        explicit IsoParametricNeighbourFinder(Mesh* mesh);
        
        // Returns a IsoParametricNodeGraph object that contains all the nodes that are within the graphDepth
        // of the node. Contains properties to get node and dof graphs map pointers.
        IsoParametricNodeGraph getIsoParametricNodeGraph(Node* node, int graphDepth);
        
        ~IsoParametricNeighbourFinder();
        
    private:
        map<vector<double>, Node*> *_parametricCoordinatesToNodeMap;
        
        Mesh* _mesh;
        
        // Maps all the parametric coordinates of mesh to the node pointer they belong to.
        map<vector<double>, Node*>* _createParametricCoordinatesToNodeMap();

    };

} // Discretization

#endif //UNTITLED_ISOPARAMETRICNEIGHBOURFINDER_H
