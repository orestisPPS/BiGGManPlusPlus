//
// Created by hal9000 on 12/17/22.
//

#ifndef UNTITLED_MESHSPECS_H
#define UNTITLED_MESHSPECS_H

#include <map>
#include "../PositioningInSpace/DirectionsPositions.h"
using namespace PositioningInSpace;
using namespace std;

namespace StructuredMeshGenerator {
    
    class MeshSpecs {
    public :
        MeshSpecs(map<Direction, unsigned> &nodesPerDirection,
                  double templateStepOne,
                  double templateRotAngle,
                  double templateShearOne);
        
        MeshSpecs(map<Direction, unsigned> &nodesPerDirection,
                  double templateStepOne, double templateStepTwo,
                  double templateRotAngle,
                  double templateShearOne, double templateShearTwo);
        
        MeshSpecs(map<Direction, unsigned> &nodesPerDirection,
                  double templateStepOne, double templateStepTwo, double templateStepThree,
                  double templateRotAngle,
                  double templateShearOne, double templateShearTwo, double templateShearThree);
        
        map<Direction, unsigned> nodesPerDirection;
        double templateStepOne, templateStepTwo, templateStepThree,
               templateRotAngle,
               templateShearOne, templateShearTwo, templateShearThree;
        };
    };

#endif //UNTITLED_MESHSPECS_H
