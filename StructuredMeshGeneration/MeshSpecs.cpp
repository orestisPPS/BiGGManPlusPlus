//
// Created by hal9000 on 12/17/22.
//

#include "MeshSpecs.h"

namespace StructuredMeshGenerator {
    
        MeshSpecs::MeshSpecs(map<Direction, unsigned> &nodesPerDirection,
                            double templateStepOne) {
            this->nodesPerDirection = nodesPerDirection;
            nodesPerDirection[Direction::Two] = 1;
            nodesPerDirection[Direction::Three] = 1;
            this->templateStepOne = templateStepOne;
            this->templateStepTwo = 0;
            this->templateStepThree = 0;
            this->templateRotAngleOne = 0;
            this->templateRotAngleTwo = 0;
            this->templateRotAngleThree = 0;
            this->templateShearOne = 0;
            this->templateShearTwo = 0;
            this->templateShearThree = 0;
            this->dimensions = 1;
        }
    
        MeshSpecs::MeshSpecs(map<Direction, unsigned> &nodesPerDirection,
                            double templateStepOne, double templateStepTwo,
                            double templateRotAngle,
                            double templateShearOne, double templateShearTwo) {
            this->nodesPerDirection = nodesPerDirection;
            nodesPerDirection[Direction::Three] = 1;
            this->templateStepOne = templateStepOne;
            this->templateStepTwo = templateStepTwo;
            this->templateStepThree = 0;
            this->templateRotAngleOne = templateRotAngle;
            this->templateRotAngleTwo = 0;
            this->templateRotAngleThree = 0;
            this->templateShearOne = templateShearOne;
            this->templateShearTwo = templateShearTwo;
            this->templateShearThree = 0;
            this->dimensions = 2;
        }
        
        MeshSpecs::MeshSpecs(map<Direction, unsigned> &nodesPerDirection,
                            double templateStepOne, double templateStepTwo, double templateStepThree,
                            double templateRotAngleOne, double templateRotAngleTwo, double templateRotAngleThree,
                            double templateShearOne, double templateShearTwo, double templateShearThree) {
            this->nodesPerDirection = nodesPerDirection;
            this->templateStepOne = templateStepOne;
            this->templateStepTwo = templateStepTwo;
            this->templateStepThree = templateStepThree;
            this->templateRotAngleOne = templateRotAngleOne;
            this->templateRotAngleTwo = templateRotAngleTwo;
            this->templateRotAngleThree = templateRotAngleThree;
            this->templateShearOne = templateShearOne;
            this->templateShearTwo = templateShearTwo;
            this->templateShearThree = templateShearThree;
            this->dimensions = 3;
        }
    

} // StructuredMeshGenerator