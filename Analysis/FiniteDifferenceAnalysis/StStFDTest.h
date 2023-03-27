//
// Created by hal9000 on 3/13/23.
//

#ifndef UNTITLED_STSTFDTEST_H
#define UNTITLED_STSTFDTEST_H

#include "SteadyStateFiniteDifferenceAnalysis.h"
#include "../../StructuredMeshGeneration/MeshPreProcessor.h"
#include "../../StructuredMeshGeneration/MeshSpecs.h"


namespace NumericalAnalysis {
    

    class StStFDTest {
public:
        StStFDTest();
        static Mesh* createMesh();
        static PartialDifferentialEquation* createPDE();
        static DomainBoundaryConditions* createBC();
        static Field_DOFType* createDOF();
        static FDSchemeSpecs* createSchemeSpecs();
    };

} // NumericalAnalysis

#endif //UNTITLED_STSTFDTEST_H
