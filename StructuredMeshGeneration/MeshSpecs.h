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
        MeshSpecs(map<Direction, short unsigned> &nodesPerDirection,
                  double templateStepOne, short unsigned metricsOrder = 2);
        
        MeshSpecs(map<Direction, short unsigned> &nodesPerDirection,
                  double templateStepOne, double templateStepTwo,
                  double templateRotAngle,
                  double templateShearOne, double templateShearTwo, short unsigned metricsOrder = 2);
        
        MeshSpecs(map<Direction, short unsigned> &nodesPerDirection,
                  double templateStepOne, double templateStepTwo, double templateStepThree,
                  double templateRotAngleOne, double templateRotAngleTwo, double templateRotAngleThree,
                  double templateShearOne, double templateShearTwo, double templateShearThree, short unsigned metricsOrder = 2);
        
        short unsigned dimensions, metricsOrder;
        map<Direction, short unsigned>& nodesPerDirection;
        double templateStepOne, templateStepTwo, templateStepThree,
               templateRotAngleOne, templateRotAngleTwo, templateRotAngleThree,
               templateShearOne, templateShearTwo, templateShearThree;
        };
    };

#endif //UNTITLED_MESHSPECS_H
