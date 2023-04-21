//
// Created by hal9000 on 1/28/23.
//

#ifndef UNTITLED_FINITEDIFFERENCESCHEME_H
#define UNTITLED_FINITEDIFFERENCESCHEME_H

#include "FDSchemeComponent.h"
#include "../../Discretization/Node/Node.h"
using namespace Discretization;


namespace LinearAlgebra {
    

    class FiniteDifferenceScheme {
    
    public:
        FiniteDifferenceScheme();
        
        ~FiniteDifferenceScheme();
        
        FDSchemeComponent* getComponentOfDerivativeOrder(unsigned derivativeOrder);
        
        map<Position, double>* schemeValuesAtAllPositions;
        
        void addFirstDerivativeComponentAtDirection(FDSchemeType schemeType, unsigned errorOrder,
                                         map<int, double>* functionValues, map<int, double>* weights, double stepSize);
        
    private:
        //  derivativeOrder     Direction    point index   weight
        map<short unsigned, map<Direction, map<unsigned , tuple<double, double, Node*>>>> _weightStepFractionNodeMap;

        

    };

} // LinearAlgebra

#endif //UNTITLED_FINITEDIFFERENCESCHEME_H
