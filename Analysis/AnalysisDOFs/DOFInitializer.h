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
        
        vector<DegreeOfFreedom*> *freeDegreesOfFreedom;
        vector<DegreeOfFreedom*> *boundedDegreesOfFreedom;
        vector<tuple<DegreeOfFreedom*, double>> *fluxDegreesOfFreedom;
        vector<DegreeOfFreedom*> *totalDegreesOfFreedom;
        map<unsigned, DegreeOfFreedom*> *totalDegreesOfFreedomMap;
        map<DegreeOfFreedom*, unsigned> *totalDegreesOfFreedomMapInverse;
        
    private:
        
        list<DegreeOfFreedom*> *_freeDegreesOfFreedomList;
        list<DegreeOfFreedom*> *_boundedDegreesOfFreedomList;
        list<tuple<DegreeOfFreedom*, double>> *_fluxDegreesOfFreedomList;
        list<DegreeOfFreedom*> *_totalDegreesOfFreedomList;
        
        void _initiateInternalNodeDOFs(Mesh *mesh, Field_DOFType* degreesOfFreedom) const;
        
        void _initiateBoundaryNodeDOFWithHomogenousBC(Mesh *mesh, Field_DOFType *problemDOFTypes,
                                                      DomainBoundaryConditions *domainBoundaryConditions) const;
        
        void _initiateBoundaryNodeDOFWithNonHomogenousBC(Mesh *mesh, Field_DOFType *problemDOFTypes,
                                                         DomainBoundaryConditions *domainBoundaryConditions) const;
        
        void _removeDuplicatesAndDelete(Mesh *mesh) const;
        
        void _assignDOFIDs() const;
        
        void _reconstructTotalDOFList(Mesh* mesh) const;

        void _assignDOFToNodes(Mesh *mesh) const;
        
        void _createTotalDOFDataStructures(Mesh *mesh) const;
        
        static void _listPtrToVectorPtr(vector<DegreeOfFreedom*> *vector, list<DegreeOfFreedom*> *list) ;
        

        
        //TODO: Implement initial conditions. Check if there is meaning in domain  initial conditions as a mathematical object.
        void applyInitialConditions(list<DegreeOfFreedom*>);
    };

} // DOFInitializer

#endif //UNTITLED_DOFINITIALIZER_H
