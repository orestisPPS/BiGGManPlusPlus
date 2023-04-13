//
// Created by hal9000 on 1/28/23.
//

#include "FDSchemeSpecs.h"



namespace LinearAlgebra {
    
    FDSchemeSpecs::FDSchemeSpecs(map<Direction,tuple<FiniteDifferenceSchemeType, int>> schemeTypeAndOrderAtDirection,
                                 SpaceEntityType &space) :
    schemeTypeAndOrderAtDirection(std::move(schemeTypeAndOrderAtDirection)){
        checkInput();
    }
    
    void FDSchemeSpecs::checkInput() {
        //Check if the order of the scheme is between the supported values (1,2,3,4,5)
        for (auto &direction : schemeTypeAndOrderAtDirection) {
            if (get<1>(direction.second) < 1 || get<1>(direction.second) > 5)
                throw std::invalid_argument("The order of the scheme must be between 1 and 5");
        }
        
        //Check if the forward difference scheme is defined for time
        if (schemeTypeAndOrderAtDirection.count(Time) != 0 && get<0>(schemeTypeAndOrderAtDirection[Time]) == Forward)
            throw std::invalid_argument("Forward scheme is not defined for time. This is a solver not a predictor");
        
        //Check if the order of the central difference scheme is even
        for (auto &direction : schemeTypeAndOrderAtDirection) {
            if (get<0>(direction.second) == Central && get<1>(direction.second) % 2 != 0)
                throw std::invalid_argument("Central scheme must have an even order");
        }
        
        //Check if the order of the scheme is consistent across all spatial directions (except time)
        auto timesChanged = 0;
        auto order = 0;
        for (auto &direction : schemeTypeAndOrderAtDirection) {
            if (timesChanged == 0) {
                order = get<1>(direction.second);
                timesChanged++;
            } else {
                if (order != get<1>(direction.second) && direction.first != Time)
                    cout<<"WARNING! The order of the scheme is not consistent across all spatial directions."<<endl;
            }
        }
    }
    
} // LinearAlgebra