//
// Created by hal9000 on 1/31/23.
//

#ifndef UNTITLED_DEGREEOFFREEDOMID_H
#define UNTITLED_DEGREEOFFREEDOMID_H


namespace DegreesOfFreedom {
    
    enum ConstraintType{
        Fixed,
        Free
    };

    class DegreeOfFreedomID {
    public:
        explicit DegreeOfFreedomID(ConstraintType type);
        
        ~DegreeOfFreedomID();
        
        bool operator == (const DegreeOfFreedomID& dof) const;

        bool operator != (const DegreeOfFreedomID& dof) const;
        
        //Pointer to the value of the degree of freedom
        unsigned int* value;

        //Constant reference to an enum that indicates whether the degree of freedom is
        // fixed (Dirichlet BC), flux (Neumann BC), or free.
        const ConstraintType& constraintType();
        
    private:
        //Enum to indicate whether the degree of freedom is fixed (Dirichlet BC), flux (Neumann BC), or free. 
        ConstraintType _constraintType;

    };
    

} // DegreesOfFreedom

#endif //UNTITLED_DEGREEOFFREEDOMID_H
