//
// Created by hal9000 on 1/28/23.
//

#include "FDSchemeSpecs.h"



namespace LinearAlgebra {
    
    FDSchemeSpecs::FDSchemeSpecs(map<Direction,tuple<FDSchemeType, int>> schemeTypeAndOrderAtDirection,
                                 SpaceEntityType &space){
        schemeTypeAndOrderAtDirectionForDerivativeOrder = new map<unsigned, map<Direction, tuple<FDSchemeType, int>>>();
        schemeTypeAndOrderAtDirectionForDerivativeOrder->
        insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>(1, schemeTypeAndOrderAtDirection));
        schemeTypeAndOrderAtDirectionForDerivativeOrder->
        insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>(2, map<Direction, tuple<FDSchemeType, int>>()));
        
        checkInput();
    }
    
    FDSchemeSpecs::FDSchemeSpecs(map<Direction,tuple<FDSchemeType, int>> schemeTypeAndOrderAtDirectionFirstDerivative,
                                 map<Direction,tuple<FDSchemeType, int>> schemeTypeAndOrderAtDirectionSecondDerivative,
                                 SpaceEntityType &space, bool diagonalTermsCalculated){
        schemeTypeAndOrderAtDirectionForDerivativeOrder = new map<unsigned, map<Direction, tuple<FDSchemeType, int>>>();
        schemeTypeAndOrderAtDirectionForDerivativeOrder->
        insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>(1, schemeTypeAndOrderAtDirectionFirstDerivative));
        schemeTypeAndOrderAtDirectionForDerivativeOrder->
        insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>(2, schemeTypeAndOrderAtDirectionSecondDerivative));
        
        checkInput();
    }
    
    FDSchemeSpecs::FDSchemeSpecs(FDSchemeType firstDerivativeSchemeType, unsigned firstDerivativeOrder,
                                 SpaceEntityType &space) {
        schemeTypeAndOrderAtDirectionForDerivativeOrder = new map<unsigned, map<Direction, tuple<FDSchemeType, int>>>();
        vector<Direction> directions;
        switch (space) {
            case Axis:
                directions = {One};
                break;
            case Plane:
                directions = {One, Two};
                break;
            case Volume:
                directions = {One, Two, Three};
                break;
            default:
                throw std::invalid_argument("Invalid space");
        }
        
        schemeTypeAndOrderAtDirectionForDerivativeOrder->insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>
                                                                        (1, map<Direction, tuple<FDSchemeType, int>>()));
        for (auto &direction : directions) {
            schemeTypeAndOrderAtDirectionForDerivativeOrder->at(1).insert(pair<Direction, tuple<FDSchemeType, int>>(
                    direction,make_tuple(firstDerivativeSchemeType,firstDerivativeOrder)));
        }
    }
    
    FDSchemeSpecs::FDSchemeSpecs(FDSchemeType firstDerivativeSchemeType, unsigned firstDerivativeOrder,
                                 FDSchemeType secondDerivativeSchemeType, unsigned secondDerivativeOrder,
                                 SpaceEntityType &space, bool diagonalTermsCalculated) {
        schemeTypeAndOrderAtDirectionForDerivativeOrder = new map<unsigned, map<Direction, tuple<FDSchemeType, int>>>();
        map<Direction, tuple<FDSchemeType, int>> schemeTypeAndOrderAtDirectionFirstDerivative;
        vector<Direction> directions;
        switch (space) {
            case Axis:
                directions = {One};
                break;
            case Plane:
                directions = {One, Two};
                break;
            case Volume:
                directions = {One, Two, Three};
                break;
            default:
                throw std::invalid_argument("Invalid space");
        }

        schemeTypeAndOrderAtDirectionForDerivativeOrder->insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>
                                                                        (1, map<Direction, tuple<FDSchemeType, int>>()));
        schemeTypeAndOrderAtDirectionForDerivativeOrder->insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>
                                                                        (2, map<Direction, tuple<FDSchemeType, int>>()));
        for (auto &direction : directions) {
            schemeTypeAndOrderAtDirectionForDerivativeOrder->at(1).insert(pair<Direction, tuple<FDSchemeType, int>>(
                    direction,make_tuple(firstDerivativeSchemeType,firstDerivativeOrder)));
            schemeTypeAndOrderAtDirectionForDerivativeOrder->at(2).insert(pair<Direction, tuple<FDSchemeType, int>>(
                    direction,make_tuple(secondDerivativeSchemeType,secondDerivativeOrder)));
        }
    }

    
    void FDSchemeSpecs::checkInput() {
        for (auto &derivativeOrder : *schemeTypeAndOrderAtDirectionForDerivativeOrder) {
            //Check if the order of the scheme is between the supported values (1,2,3,4,5)
            for (auto &direction : derivativeOrder.second) {
                if (get<1>(direction.second) < 1 || get<1>(direction.second) > 5)
                    throw std::invalid_argument("The order of the scheme must be between 1 and 5");
            }

            //Check if the forward difference scheme is defined for time
            for (auto &direction : derivativeOrder.second) {
                if (get<0>(direction.second) == Forward && direction.first == Time)
                    throw std::invalid_argument("Forward scheme is not defined for time. This is a solver not a time machine.");
            }   

            //Check if the order of the central difference scheme is even
            for (auto &direction : derivativeOrder.second) {
                if (get<0>(direction.second) == Central &&
                    get<1>(direction.second) % 2 != 0 && get<1>(direction.second) <= 6)
                    throw std::invalid_argument("Central scheme must have an even order and be less than 6");

            }

            //Check if the order of the scheme is consistent across all spatial directions (except time)
            auto timesChanged = 0;
            auto order = 0;
            for (auto &direction : derivativeOrder.second) {
                if (timesChanged == 0) {
                    order = get<1>(direction.second);
                    timesChanged++;
                } else {
                    if (order != get<1>(direction.second) && direction.first != Time)
                        cout<<"WARNING! The order of the scheme is not consistent across all spatial directions."<<endl;
                }
            }
        }
    }
}// LinearAlgebra