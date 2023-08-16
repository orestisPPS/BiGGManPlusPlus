//
// Created by hal9000 on 7/15/23.
//

#ifndef UNTITLED_STEADYSTATE3DNEUMANN_H
#define UNTITLED_STEADYSTATE3DNEUMANN_H

#include "../Analysis/FiniteDifferenceAnalysis/SteadyStateFiniteDifferenceAnalysis.h"
#include "../StructuredMeshGeneration/MeshFactory.h"
#include "../StructuredMeshGeneration/MeshSpecs.h"
#include "../StructuredMeshGeneration/DomainBoundaryFactory.h"
#include "../LinearAlgebra/Solvers/Iterative/GradientBasedIterative/ConjugateGradientSolver.h"


namespace Tests {

    class SteadyState3DNeumann {
    public:
        SteadyState3DNeumann();
    };

} // Tests

#endif //UNTITLED_STEADYSTATE3DNEUMANN_H
