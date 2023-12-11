//
// Created by hal9000 on 3/12/23.
//

#ifndef UNTITLED_ANALYSISDEGREESOFFREEDOM_H
#define UNTITLED_ANALYSISDEGREESOFFREEDOM_H
#include <tuple>
#include "../../MathematicalEntities/MathematicalProblem/SteadyStateMathematicalProblem.h"
#include "../../DegreesOfFreedom/DegreeOfFreedom.h"
#include "DOFInitializer.h"

namespace NumericalAnalysis {

    class AnalysisDegreesOfFreedom {
    public:
        AnalysisDegreesOfFreedom(const shared_ptr<Mesh>& mesh, const shared_ptr<DomainBoundaryConditions>&domainBoundaryConditions,
                                 Field_DOFType* degreesOfFreedom);
        
        ~AnalysisDegreesOfFreedom();
        shared_ptr<vector<DegreeOfFreedom*>> totalDegreesOfFreedom;

        shared_ptr<vector<DegreeOfFreedom*>> freeDegreesOfFreedom;

        shared_ptr<vector<DegreeOfFreedom*>> fixedDegreesOfFreedom;
        
        shared_ptr<vector<DegreeOfFreedom*>> internalDegreesOfFreedom;
        
        shared_ptr<unordered_map<DegreeOfFreedom*, double>> fluxDegreesOfFreedom;
        
        shared_ptr<unordered_map<unsigned, DegreeOfFreedom*>> totalDegreesOfFreedomMap; 
        
        shared_ptr<unordered_map<DegreeOfFreedom*, unsigned>> totalDegreesOfFreedomMapInverse;

        shared_ptr<unsigned int> numberOfFixedDOF;

        shared_ptr<unsigned int> numberOfFreeDOF;

        shared_ptr<unsigned int> numberOfDOF;
        
        void printDOFCount() const;
        
    private:
        
        void _deallocateDegreesOfFreedom() const;
    };

} // NumericalAnalysis

#endif //UNTITLED_ANALYSISDEGREESOFFREEDOM_H
