//
// Created by hal9000 on 3/12/23.
//

#ifndef UNTITLED_ANALYSISDEGREESOFFREEDOM_H
#define UNTITLED_ANALYSISDEGREESOFFREEDOM_H
#include <tuple>
#include "../../MathematicalProblem/SteadyStateMathematicalProblem.h"
#include "../../DegreesOfFreedom/DegreeOfFreedom.h"
#include "DOFInitializer.h"

namespace NumericalAnalysis {

    class AnalysisDegreesOfFreedom {
    public:
        AnalysisDegreesOfFreedom(Mesh *mesh, DomainBoundaryConditions *domainBoundaryConditions,
                                 Field_DOFType* degreesOfFreedom);
        
        ~AnalysisDegreesOfFreedom();
        
        vector<DegreeOfFreedom*> *totalDegreesOfFreedom;

        vector<DegreeOfFreedom*> *freeDegreesOfFreedom;

        vector<DegreeOfFreedom*> *fixedDegreesOfFreedom;

        vector<tuple<DegreeOfFreedom*, double>> *fluxDegreesOfFreedom;
        
        map<unsigned, DegreeOfFreedom*> *totalDegreesOfFreedomMap;
        
        map<DegreeOfFreedom*, unsigned> *totalDegreesOfFreedomMapInverse;
        
        void printDOFCount() const;
        
    private:
        
        void _deallocateDegreesOfFreedom() const;
    };

} // NumericalAnalysis

#endif //UNTITLED_ANALYSISDEGREESOFFREEDOM_H
