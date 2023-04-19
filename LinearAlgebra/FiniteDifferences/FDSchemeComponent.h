//
// Created by hal9000 on 4/15/23.
//

#ifndef UNTITLED_FDSCHEMECOMPONENT_H
#define UNTITLED_FDSCHEMECOMPONENT_H

#include <map>
#include <stdexcept>

#include "FDScheme.h"

using namespace std;
namespace LinearAlgebra {

    class FDSchemeComponent {
    public:
        
        // Gets the 
        FDSchemeComponent(FDSchemeType schemeType, unsigned errorOrder,
                          map<int, double>* functionValues, map<int, double>* weights, double stepSize);

        FDSchemeComponent(FDSchemeType schemeType, unsigned errorOrder,
                          map<int, double>* functionValues, map<int, double>* weights, map<int, double>* stepFractions);
        
        ~FDSchemeComponent();

        FDSchemeType schemeType;
        
        unsigned errorOrder;
        
        map<int, double>* functionValues;
        
        map<int, double>* weights;
        
        map<int, double>* stepFractions;
        
        map<int, double>* values;
        
    private:
       static void checkInput(map<int, double>* functionValues, map<int, double>* weights);
       
       static void checkInput(map<int, double>* functionValues, map<int, double>* weights, map<int, double>* stepFractions);
       
       map <int, double>* calculateValues() const;
    };

} // LinearAlgebra

#endif //UNTITLED_FDSCHEMECOMPONENT_H
