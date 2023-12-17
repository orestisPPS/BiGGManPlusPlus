//
// Created by hal9000 on 4/12/23.
//

#include "Metrics.h"


namespace Discretization{
    
        Metrics::Metrics(Node* node, unsigned dimensions) {
            this->node = node;
            covariantBaseVectors = make_unique<map<Direction, NumericalVector<double>>>();
            contravariantBaseVectors = make_unique<map<Direction, NumericalVector<double>>>();
            covariantTensor = make_unique<NumericalMatrix<double>>(dimensions, dimensions);
            contravariantTensor = make_unique<NumericalMatrix<double>>(dimensions, dimensions);
            
            covariantTensor = make_unique<NumericalMatrix<double>>(dimensions, dimensions);
            contravariantTensor = make_unique<NumericalMatrix<double>>(dimensions, dimensions);
            jacobian = make_unique<double>();
        }
        
        void Metrics::calculateCovariantTensor() const {
            auto n = covariantBaseVectors->size();
            covariantTensor->dataStorage->initializeElementAssignment();
            for (auto i = 0; i < n; i++) {
                for (auto j = 0; j < n; j++) {
                    auto gi = covariantBaseVectors->at(unsignedToSpatialDirection[i]);
                    auto gj = covariantBaseVectors->at(unsignedToSpatialDirection[j]);
                    auto gij = gi.dotProduct(gj);
                    covariantTensor->setElement(i, j, gij);
                }
            }
            covariantTensor->dataStorage->finalizeElementAssignment();
        }
        
        void Metrics::calculateContravariantTensor() const {
            auto n = contravariantBaseVectors->size();
            contravariantTensor->dataStorage->initializeElementAssignment();
            for (auto i = 0; i < n; i++) {
                for (auto j = 0; j < n; j++) {
                    auto gi = contravariantBaseVectors->at(unsignedToSpatialDirection[i]);
                    auto gj = contravariantBaseVectors->at(unsignedToSpatialDirection[j]);
                    auto gij = gi.dotProduct(gj);
                    contravariantTensor->setElement(i, j, gij);
                }
            }
            contravariantTensor->dataStorage->finalizeElementAssignment();
        }
    
    } // Discretization

