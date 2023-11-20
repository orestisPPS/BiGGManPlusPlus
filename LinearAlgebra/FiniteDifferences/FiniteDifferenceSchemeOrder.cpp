//
// Created by hal9000 on 1/28/23.
//

#include "FiniteDifferenceSchemeOrder.h"



namespace LinearAlgebra {
    
    FiniteDifferenceSchemeOrder::FiniteDifferenceSchemeOrder(const map<Direction,tuple<FDSchemeType, int>>& schemeTypeAndOrderAtDirection){
        schemeTypeAndOrderAtDirectionForDerivativeOrder = new map<unsigned, map<Direction, tuple<FDSchemeType, int>>>();
        schemeTypeAndOrderAtDirectionForDerivativeOrder->
        insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>(1, schemeTypeAndOrderAtDirection));
        schemeTypeAndOrderAtDirectionForDerivativeOrder->
        insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>(2, map<Direction, tuple<FDSchemeType, int>>()));
        
        checkInput();
        schemeTypeFixed = true;
    }
    
    FiniteDifferenceSchemeOrder::FiniteDifferenceSchemeOrder(const map<Direction,tuple<FDSchemeType, int>>& schemeTypeAndOrderAtDirectionFirstDerivative,
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
    
    FiniteDifferenceSchemeOrder::FiniteDifferenceSchemeOrder(FDSchemeType firstDerivativeSchemeType, unsigned firstDerivativeOrder,
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
    
    FiniteDifferenceSchemeOrder::FiniteDifferenceSchemeOrder(unsigned short firstDerivativeErrorOrder, const vector<Direction> &directions) {
        schemeTypeAndOrderAtDirectionForDerivativeOrder = new map<unsigned, map<Direction, tuple<FDSchemeType, int>>>();
        //Insert Specs for first derivative
        schemeTypeAndOrderAtDirectionForDerivativeOrder->insert(pair<unsigned, map<Direction, tuple<FDSchemeType, int>>>
                                                                        (1, map<Direction, tuple<FDSchemeType, int>>()));
        for (auto &direction : directions) {
            schemeTypeAndOrderAtDirectionForDerivativeOrder->at(1).insert(pair<Direction, tuple<FDSchemeType, int>>(
                    direction,make_tuple(FDSchemeType::Arbitrary,firstDerivativeErrorOrder)));
        }
        schemeTypeFixed = true;
    }
    
    FiniteDifferenceSchemeOrder::FiniteDifferenceSchemeOrder(FDSchemeType firstDerivativeSchemeType, unsigned firstDerivativeOrder,
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
    
    FiniteDifferenceSchemeOrder::FiniteDifferenceSchemeOrder(unsigned firstDerivativeOrder, unsigned secondDerivativeOrder,
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
    
    FiniteDifferenceSchemeOrder::~FiniteDifferenceSchemeOrder() {
        delete schemeTypeAndOrderAtDirectionForDerivativeOrder;
    }

    
    void FiniteDifferenceSchemeOrder::checkInput() const {
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

    unsigned FiniteDifferenceSchemeOrder::getErrorOrderOfSchemeTypeForDerivative(unsigned int derivativeOrder) const {
        if (!schemeTypeFixed)
            return get<1>(schemeTypeAndOrderAtDirectionForDerivativeOrder->at(derivativeOrder).at(One));
        else
            throw std::invalid_argument(" The scheme type is fixed. Use this method only if the error order is fixed"
                                        " and the scheme type varies across the domain depending on the neighbours in order"
                                        " to achieve the desired error order.");
        
    }

    unsigned FiniteDifferenceSchemeOrder::getErrorForDerivativeOfArbitraryScheme(unsigned int derivativeOrder) const {
        if (schemeTypeFixed){
            bool found = false;
            for (auto &direction : schemeTypeAndOrderAtDirectionForDerivativeOrder->at(derivativeOrder))
                if (get<0>(direction.second) == Arbitrary)
                    found = true;
            if (found)
                return get<1>(schemeTypeAndOrderAtDirectionForDerivativeOrder->at(derivativeOrder).at(One));
            else
                throw std::invalid_argument(" The scheme type is fixed. Use this method only if the error order is fixed"
                                            " and the scheme type varies across the domain depending on the neighbours in order"
                                            " to achieve the desired error order.");
        }
        else{
            throw std::invalid_argument(" The scheme type is not fixed. Use this method only if the error order is fixed"
                                        " and the scheme type varies across the domain depending on the neighbours in order"
                                        " to achieve the desired error order.");
        }
    }
    
}// LinearAlgebra