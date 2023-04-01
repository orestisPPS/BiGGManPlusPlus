//
// Created by hal9000 on 4/1/23.
//

#ifndef UNTITLED_ISOPARAMETRICCURVES_H
#define UNTITLED_ISOPARAMETRICCURVES_H

#include "../Discretization/Mesh/Mesh.h"

using namespace PositioningInSpace;
using namespace Discretization;

namespace PositioningInSpace {

    class IsoParametricCurves {
        public:
        IsoParametricCurves(Mesh* mesh);
        
        //Map of iso-parametric curves containing all the nodes that belong to the same parametric coordinate axis.
        //Key: Direction (One (ξ), Two (η), Three(ζ)
        //Value: Iso-parametric curve map at the key direction
        //Key: Parametric coordinate at the iso-parametric curve
        //Value: Node pointer with the node at the parametric coordinate value
        map<Direction, map<double, Node*>> isoParametricCurves;
        
    };

} // PositioningInSpace

#endif //UNTITLED_ISOPARAMETRICCURVES_H
