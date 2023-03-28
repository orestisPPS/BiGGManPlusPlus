//
// Created by hal9000 on 3/12/23.
//

#ifndef UNTITLED_ANALYSISDEGREESOFFREEDOM_H
#define UNTITLED_ANALYSISDEGREESOFFREEDOM_H
#include <tuple>
#include "../../MathematicalProblem/SteadyStateMathematicalProblem.h"
#include "../../DegreesOfFreedom/DegreeOfFreedom.h"
#include "DOFInitializer.h"
#include "../../Discretization/Mesh/Mesh.h"

namespace NumericalAnalysis {

    class AnalysisDegreesOfFreedom {
    public:
        AnalysisDegreesOfFreedom(Mesh *mesh, DomainBoundaryConditions *domainBoundaryConditions,
                                 Field_DOFType* degreesOfFreedom);
        
        ~AnalysisDegreesOfFreedom();
        
        map<unsigned*, vector<DegreeOfFreedom*>*> *nodeDoFMap; 
        
        vector<DegreeOfFreedom*> *totalDegreesOfFreedom;

        vector<DegreeOfFreedom*> *freeDegreesOfFreedom;

        vector<DegreeOfFreedom*> *boundedDegreesOfFreedom;

        vector<tuple<DegreeOfFreedom*, double>> *fluxDegreesOfFreedom;
        
        void printDOFCount() const;
        
    private:
        
        map<unsigned*, vector<DegreeOfFreedom*>> _createNodeDofMap(Mesh *mesh, Field_DOFType* degreesOfFreedom);
        
        void _deallocateDegreesOfFreedom() const;

        
    };

} // NumericalAnalysis

#endif //UNTITLED_ANALYSISDEGREESOFFREEDOM_H
