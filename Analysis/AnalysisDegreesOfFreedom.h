//
// Created by hal9000 on 3/12/23.
//

#ifndef UNTITLED_ANALYSISDEGREESOFFREEDOM_H
#define UNTITLED_ANALYSISDEGREESOFFREEDOM_H
#include <tuple>
#include "../MathematicalProblem/SteadyStateMathematicalProblem.h"
#include "../DegreesOfFreedom/DegreeOfFreedom.h"
#include "DOFInitializer.h"
#include "../Discretization/Mesh/Mesh.h"

namespace Analysis {

    class AnalysisDegreesOfFreedom {
    public:
        AnalysisDegreesOfFreedom();
        
        ~AnalysisDegreesOfFreedom();
        
        list<DegreeOfFreedom*> *totalDegreesOfFreedom;
        
        list<DegreeOfFreedom*> *freeDegreesOfFreedom;
        
        list<DegreeOfFreedom*> *boundedDegreesOfFreedom;
        
        list<tuple<DegreeOfFreedom*, double>> *fluxDegreesOfFreedom;
        
        void _initiateDegreesOfFreedom(Mesh *mesh, DomainBoundaryConditions *domainBoundaryConditions,
                                       struct Field_DOFType* degreesOfFreedom);
        void printDOFCount() const;
        
    private:
        
        void _deallocateDegreesOfFreedom() const;

        
    };

} // Analysis

#endif //UNTITLED_ANALYSISDEGREESOFFREEDOM_H
