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
                  double templateStepOne, unsigned short metricsOrder = 2);
        
        MeshSpecs(map<Direction, unsigned> &nodesPerDirection,
                  double templateStepOne, double templateStepTwo,
                  double templateRotAngle,
                  double templateShearOne, double templateShearTwo, unsigned short metricsOrder = 2);
        
        MeshSpecs(map<Direction, unsigned> &nodesPerDirection,
                  double templateStepOne, double templateStepTwo, double templateStepThree,
                  double templateRotAngleOne, double templateRotAngleTwo, double templateRotAngleThree,
                  double templateShearOne, double templateShearTwo, double templateShearThree, unsigned short metricsOrder = 2);
        
        unsigned dimensions, metricsOrder;
        map<Direction, unsigned>& nodesPerDirection;
        double templateStepOne, templateStepTwo, templateStepThree,
               templateRotAngleOne, templateRotAngleTwo, templateRotAngleThree,
               templateShearOne, templateShearTwo, templateShearThree;
        };
    };

#endif //UNTITLED_MESHSPECS_H
