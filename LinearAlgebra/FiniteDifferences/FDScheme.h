//
// Created by hal9000 on 1/28/23.
//

#ifndef UNTITLED_FDSCHEME_H
#define UNTITLED_FDSCHEME_H

#include "FDSchemeComponent.h"


namespace LinearAlgebra {
    

    class FDScheme {
    
    public:
        explicit FDScheme(map<unsigned, FDSchemeComponent*>* components);
        
        ~FDScheme();
        
        FDSchemeComponent* getComponentOfDerivativeOrder(unsigned derivativeOrder);
        
        map<Position, double>* schemeValuesAtAllPositions;
        
    private:
        map<unsigned, FDSchemeComponent*>* _components;
        
        map<Position, double>* calculateSchemeValuesAtAllPositions();
        
        

    };

} // LinearAlgebra

#endif //UNTITLED_FDSCHEME_H
