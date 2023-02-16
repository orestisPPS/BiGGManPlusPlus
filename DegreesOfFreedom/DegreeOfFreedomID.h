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
        DegreeOfFreedomID();
        unsigned int _id;
        ConstraintType _constraintType;
    private:

    };
    

} // DegreesOfFreedom

#endif //UNTITLED_DEGREEOFFREEDOMID_H
