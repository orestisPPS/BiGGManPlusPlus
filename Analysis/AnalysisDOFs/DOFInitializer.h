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
        DOFInitializer(const shared_ptr<Mesh>& mesh, const shared_ptr<DomainBoundaryConditions>&domainBoundaryConditions, struct Field_DOFType* degreesOfFreedom);

        shared_ptr<vector<DegreeOfFreedom*>> totalDegreesOfFreedom;
        
        shared_ptr<vector<DegreeOfFreedom*>> fixedDegreesOfFreedom;

        shared_ptr<vector<DegreeOfFreedom*>> freeDegreesOfFreedom;
        
        shared_ptr<vector<DegreeOfFreedom*>> internalDegreesOfFreedom;
        
        shared_ptr<map<DegreeOfFreedom*, double>> fluxDegreesOfFreedom;

        shared_ptr<map<unsigned, DegreeOfFreedom*>> totalDegreesOfFreedomMap;

        shared_ptr<map<DegreeOfFreedom*, unsigned>> totalDegreesOfFreedomMapInverse;

        shared_ptr<unsigned int> numberOfFixedDOF;

        shared_ptr<unsigned int> numberOfFreeDOF;

        shared_ptr<unsigned int> numberOfDOF;
        
    private:
        shared_ptr<list<DegreeOfFreedom*>> _totalDegreesOfFreedomList;
        
        shared_ptr<list<DegreeOfFreedom*>> _freeDegreesOfFreedomList;
        
        shared_ptr<list<DegreeOfFreedom*>> _fixedDegreesOfFreedomList;
        
        shared_ptr<list<DegreeOfFreedom*>> _internalDegreesOfFreedomList;
        
        void _initiateInternalNodeDOFs(const shared_ptr<Mesh>& mesh, Field_DOFType* degreesOfFreedom);
        
        void _initiateBoundaryNodeDOFWithHomogenousBC(const shared_ptr<Mesh>& mesh, Field_DOFType *problemDOFTypes,
                                                      const shared_ptr<DomainBoundaryConditions>&domainBoundaryConditions) ;
        
        void _initiateBoundaryNodeDOF(const shared_ptr<Mesh> &mesh, Field_DOFType *problemDOFTypes,
                                      const shared_ptr<DomainBoundaryConditions> &domainBoundaryConditions);
        
        void _assignDOFIDs() const;
        
        void _createTotalDOFList(const shared_ptr<Mesh>& mesh) const;
        
        void _createTotalDOFDataStructures(const shared_ptr<Mesh>& mesh) const;
        
        static void _listPtrToVectorPtr(const shared_ptr<list<DegreeOfFreedom*>>& listPtr, const shared_ptr<vector<DegreeOfFreedom*>>& vectorPtr) ;
        
        //TODO: Implement initial conditions. Check if there is meaning in domain  initial conditions as a mathematical object.
        void applyInitialConditions(list<DegreeOfFreedom*>);
    };

} // DOFInitializer

#endif //UNTITLED_DOFINITIALIZER_H
