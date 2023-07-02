//
// Created by hal9000 on 3/13/23.
//

#ifndef UNTITLED_STSTFDTEST_H
#define UNTITLED_STSTFDTEST_H

#include "SteadyStateFiniteDifferenceAnalysis.h"
#include "../../StructuredMeshGeneration/MeshFactory.h"
#include "../../StructuredMeshGeneration/MeshSpecs.h"
#include "../../StructuredMeshGeneration/DomainBoundaryFactory.h"
#include "../../LinearAlgebra/Solvers/Direct/SolverLUP.h"


namespace NumericalAnalysis {
    

    class StStFDTest {
public:
        StStFDTest();

        static shared_ptr<DomainBoundaryConditions> createBC(shared_ptr<Mesh> mesh);
        static Field_DOFType* createDOF();
        static shared_ptr<FDSchemeSpecs> createSchemeSpecs();
        
    };

} // NumericalAnalysis

#endif //UNTITLED_STSTFDTEST_H
