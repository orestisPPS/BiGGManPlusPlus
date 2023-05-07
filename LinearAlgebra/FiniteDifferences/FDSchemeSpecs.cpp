//
// Created by hal9000 on 1/28/23.
//

#include "FDSchemeSpecs.h"



namespace LinearAlgebra {
    
    FDSchemeSpecs::FDSchemeSpecs(const map<Direction,tuple<FDSchemeType, int>>& schemeTypeAndOrderAtDirection){
        schemeTypeAndOrderAtDirectionForDerivativeOrder = new map<unsigned, map<Direction, tuple<FDSchemeType, int>>>();
        schemeTypeAndOrderAtDirectionForDerivativeOrder->
        insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>(1, schemeTypeAndOrderAtDirection));
        schemeTypeAndOrderAtDirectionForDerivativeOrder->
        insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>(2, map<Direction, tuple<FDSchemeType, int>>()));
        
        checkInput();
        schemeTypeFixed = true;
    }
    
    FDSchemeSpecs::FDSchemeSpecs(const map<Direction,tuple<FDSchemeType, int>>& schemeTypeAndOrderAtDirectionFirstDerivative,
                                 const map<Direction,tuple<FDSchemeType, int>>& schemeTypeAndOrderAtDirectionSecondDerivative,
                                 bool diagonalTermsCalculated){
        schemeTypeAndOrderAtDirectionForDerivativeOrder = new map<unsigned, map<Direction, tuple<FDSchemeType, int>>>();
        schemeTypeAndOrderAtDirectionForDerivativeOrder->
        insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>(1, schemeTypeAndOrderAtDirectionFirstDerivative));
        schemeTypeAndOrderAtDirectionForDerivativeOrder->
        insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>(2, schemeTypeAndOrderAtDirectionSecondDerivative));
        
        checkInput();
        schemeTypeFixed = true;
    }
    
    FDSchemeSpecs::FDSchemeSpecs(FDSchemeType firstDerivativeSchemeType, unsigned firstDerivativeOrder,
                                 const vector<Direction> &directions) {
        schemeTypeAndOrderAtDirectionForDerivativeOrder = new map<unsigned, map<Direction, tuple<FDSchemeType, int>>>();
        //Insert Specs for first derivative
        schemeTypeAndOrderAtDirectionForDerivativeOrder->insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>
                                                                        (1, map<Direction, tuple<FDSchemeType, int>>()));
        for (auto &direction : directions) {
            schemeTypeAndOrderAtDirectionForDerivativeOrder->at(1).insert(pair<Direction, tuple<FDSchemeType, int>>(
                    direction,make_tuple(firstDerivativeSchemeType,firstDerivativeOrder)));
        }
        schemeTypeFixed = true;

    }
    
    FDSchemeSpecs::FDSchemeSpecs(FDSchemeType firstDerivativeSchemeType, unsigned firstDerivativeOrder,
                                 FDSchemeType secondDerivativeSchemeType, unsigned secondDerivativeOrder,
                                 const vector<Direction> &directions, bool diagonalTermsCalculated) {
        
        schemeTypeAndOrderAtDirectionForDerivativeOrder = new map<unsigned, map<Direction, tuple<FDSchemeType, int>>>();
        //Insert Specs for first derivative
        schemeTypeAndOrderAtDirectionForDerivativeOrder->insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>
                                                                        (1, map<Direction, tuple<FDSchemeType, int>>()));
        //Insert Specs for second derivative
        schemeTypeAndOrderAtDirectionForDerivativeOrder->insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>
                                                                        (2, map<Direction, tuple<FDSchemeType, int>>()));
        for (auto &direction : directions) {
            schemeTypeAndOrderAtDirectionForDerivativeOrder->at(1).insert(pair<Direction, tuple<FDSchemeType, int>>(
                    direction,make_tuple(firstDerivativeSchemeType,firstDerivativeOrder)));
            schemeTypeAndOrderAtDirectionForDerivativeOrder->at(2).insert(pair<Direction, tuple<FDSchemeType, int>>(
                    direction,make_tuple(secondDerivativeSchemeType,secondDerivativeOrder)));
        }
        schemeTypeFixed = true;
    }
    
    FDSchemeSpecs::FDSchemeSpecs(unsigned firstDerivativeOrder, unsigned secondDerivativeOrder,
                                 const vector<Direction> &directions, bool diagonalTermsCalculated) {
        schemeTypeAndOrderAtDirectionForDerivativeOrder = new map<unsigned, map<Direction, tuple<FDSchemeType, int>>>();
        //Insert Specs for first derivative
        schemeTypeAndOrderAtDirectionForDerivativeOrder->insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>
                                                                        (1, map<Direction, tuple<FDSchemeType, int>>()));
        //Insert Specs for second derivative
        schemeTypeAndOrderAtDirectionForDerivativeOrder->insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>
                                                                        (2, map<Direction, tuple<FDSchemeType, int>>()));
        for (auto &direction : directions) {
            schemeTypeAndOrderAtDirectionForDerivativeOrder->at(1).insert(pair<Direction, tuple<FDSchemeType, int>>(
                    direction,make_tuple(Arbitrary, firstDerivativeOrder)));
            schemeTypeAndOrderAtDirectionForDerivativeOrder->at(2).insert(pair<Direction, tuple<FDSchemeType, int>>(
                    direction,make_tuple(Arbitrary, secondDerivativeOrder)));
        }
        schemeTypeFixed = false;
    }
    
    FDSchemeSpecs::~FDSchemeSpecs() {
        delete schemeTypeAndOrderAtDirectionForDerivativeOrder;
    }

    
    void FDSchemeSpecs::checkInput() const {
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