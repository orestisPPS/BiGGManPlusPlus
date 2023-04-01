//
// Created by hal9000 on 4/1/23.
//

#ifndef UNTITLED_ISOPARAMETRICCURVES_H
#define UNTITLED_ISOPARAMETRICCURVES_H

#include "DirectionsPositions.h"
#include "../Discretization/Node/Node.h"

using namespace PositioningInSpace;
using namespace Discretization;

namespace PositioningInSpace {

    class IsoParametricCurves {
        
        public:
        
        explicit IsoParametricCurves(map<Direction, map<double,Node*>*>* isoParametricCurves);
        
        Node* getNodeAtParametricDirectionAndCoordinate(Direction direction, double coordinate);
        
        unsigned getNodeIdAtParametricDirectionAndCoordinate(Direction direction, double coordinate);
        
        Node* getNodeWithParametricCoordinates(vector<double>*);
        
        unsigned getNodeIdWithParametricCoordinates(vector<double>*);
        
        vector<Node*> getIsoCurveNodes(Direction direction, double constantCoordinate);
        
        vector<unsigned> getIsoCurveNodeIds(Direction direction, double constantCoordinate);
                
    private:
        //Map of iso-parametric curves containing all the nodes that belong to the same parametric coordinate axis.
        //Key: Direction (One (ξ), Two (η), Three(ζ)
        //Value: Iso-parametric curve map at the key direction
        //Key: Parametric axis coordinate
        //Value: Node pointer of the node with key coordinate
        map<Direction, map<double,Node*>*>* isoParametricCurves;
        
    };

} // PositioningInSpace

#endif //UNTITLED_ISOPARAMETRICCURVES_H
