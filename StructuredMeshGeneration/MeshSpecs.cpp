//
// Created by hal9000 on 12/17/22.
//

#include "MeshSpecs.h"

namespace StructuredMeshGenerator {
    
        MeshSpecs::MeshSpecs(map<Direction, unsigned> &nodesPerDirection,
                            double templateStepOne,
                            double templateRotAngle,
                            double templateShearOne) {
            this->nodesPerDirection = nodesPerDirection;
            this->templateStepOne = templateStepOne;
            this->templateStepTwo = 0;
            this->templateStepThree = 0;
            this->templateRotAngle = templateRotAngle;
            this->templateShearOne = templateShearOne;
            this->templateShearTwo = 0;
            this->templateShearThree = 0;
        }
    
        MeshSpecs::MeshSpecs(map<Direction, unsigned> &nodesPerDirection,
                            double templateStepOne, double templateStepTwo,
                            double templateRotAngle,
                            double templateShearOne, double templateShearTwo) {
            this->nodesPerDirection = nodesPerDirection;
            this->templateStepOne = templateStepOne;
            this->templateStepTwo = templateStepTwo;
            this->templateStepThree = 0;
            this->templateRotAngle = templateRotAngle;
            this->templateShearOne = templateShearOne;
            this->templateShearTwo = templateShearTwo;
            this->templateShearThree = 0;
        }
    
        MeshSpecs::MeshSpecs(map<Direction, unsigned> &nodesPerDirection,
                            double templateStepOne, double templateStepTwo, double templateStepThree,
                            double templateRotAngle,
                            double templateShearOne, double templateShearTwo, double templateShearThree) {
            this->nodesPerDirection = nodesPerDirection;
            this->templateStepOne = templateStepOne;
            this->templateStepTwo = templateStepTwo;
            this->templateStepThree = templateStepThree;
            this->templateRotAngle = templateRotAngle;
            this->templateShearOne = templateShearOne;
            this->templateShearTwo = templateShearTwo;
            this->templateShearThree = templateShearThree;
        }
    

} // StructuredMeshGenerator