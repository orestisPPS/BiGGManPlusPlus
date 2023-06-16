//
// Created by hal9000 on 2/18/23.
//

#ifndef UNTITLED_DOFINITIALIZER_H
#define UNTITLED_DOFINITIALIZER_H

#include "../../MathematicalProblem/SteadyStateMathematicalProblem.h"
#include "../../DegreesOfFreedom/DegreeOfFreedom.h"
#include "../../Discretization/Mesh/Mesh.h"

namespace NumericalAnalysis {

    class DOFInitializer {
    public:
        DOFInitializer(Mesh *mesh, DomainBoundaryConditions *domainBoundaryConditions, struct Field_DOFType* degreesOfFreedom);

        shared_ptr<vector<DegreeOfFreedom*>> fixedDegreesOfFreedom;

        shared_ptr<vector<tuple<DegreeOfFreedom*, double>>> fluxDegreesOfFreedom;

        shared_ptr<map<unsigned, DegreeOfFreedom*>> totalDegreesOfFreedomMap;

        shared_ptr<map<DegreeOfFreedom*, unsigned>> totalDegreesOfFreedomMapInverse;

        shared_ptr<unsigned int> numberOfFixedDOF;

        shared_ptr<unsigned int> numberOfFreeDOF;

        shared_ptr<unsigned int> numberOfDOF;
        
    private:
        
        unique_ptr<list<DegreeOfFreedom*>> _freeDegreesOfFreedomList;
        unique_ptr<list<DegreeOfFreedom*>> _fluxDegreesOfFreedomList;
        unique_ptr<list<DegreeOfFreedom*>> _boundedDegreesOfFreedomList;
        
        void _initiateInternalNodeDOFs(Mesh *mesh, Field_DOFType* degreesOfFreedom);
        
        void _initiateBoundaryNodeDOFWithHomogenousBC(Mesh *mesh, Field_DOFType *problemDOFTypes,
                                                      DomainBoundaryConditions *domainBoundaryConditions) ;
        
        void _initiateBoundaryNodeDOFWithNonHomogenousBC(Mesh *mesh, Field_DOFType *problemDOFTypes,
                                                         DomainBoundaryConditions *domainBoundaryConditions) ;
        
        void _assignDOFIDs() const;
        
        void _createTotalDOFList(Mesh* mesh) const;
        
        void _createTotalDOFDataStructures(Mesh *mesh) const;
        
        static void _listPtrToVectorPtr(vector<DegreeOfFreedom*> *vector, list<DegreeOfFreedom*> *list) ;
        
        //TODO: Implement initial conditions. Check if there is meaning in domain  initial conditions as a mathematical object.
        void applyInitialConditions(list<DegreeOfFreedom*>);
    };

} // DOFInitializer

#endif //UNTITLED_DOFINITIALIZER_H
