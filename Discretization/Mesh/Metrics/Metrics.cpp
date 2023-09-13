//
// Created by hal9000 on 4/12/23.
//

#include "Metrics.h"


namespace Discretization{
    
        Metrics::Metrics(Node* node, unsigned dimensions) {
            this->node = node;
            covariantBaseVectors = make_shared<map<Direction, NumericalVector<double>>>();
            contravariantBaseVectors = make_shared<map<Direction, NumericalVector<double>>>();
            covariantTensor = make_shared<NumericalMatrix<double>>(dimensions, dimensions);
            contravariantTensor = make_shared<NumericalMatrix<double>>(dimensions, dimensions);

/*            switch (dimensions) {
                case 1:
                    covariantBaseVectors->insert(pair<Direction, NumericalVector<double>>(One, NumericalVector<double>()));
                    contravariantBaseVectors->insert(pair<Direction, NumericalVector<double>>(One, NumericalVector<double>()));
                    break;
                case 2:
                    covariantBaseVectors->insert(pair<Direction, NumericalVector<double>>(One, NumericalVector<double>()));
                    covariantBaseVectors->insert(pair<Direction, NumericalVector<double>>(Two, NumericalVector<double>()));
                    
                    contravariantBaseVectors->insert(pair<Direction, NumericalVector<double>>(One, NumericalVector<double>()));
                    contravariantBaseVectors->insert(pair<Direction, NumericalVector<double>>(Two, NumericalVector<double>()));
                    break;
                case 3:
                    covariantBaseVectors->insert(pair<Direction, NumericalVector<double>>(One, NumericalVector<double>()));
                    covariantBaseVectors->insert(pair<Direction, NumericalVector<double>>(Two, NumericalVector<double>()));
                    covariantBaseVectors->insert(pair<Direction, NumericalVector<double>>(Three, NumericalVector<double>()));
                    
                    contravariantBaseVectors->insert(pair<Direction, NumericalVector<double>>(One, NumericalVector<double>()));
                    contravariantBaseVectors->insert(pair<Direction, NumericalVector<double>>(Two, NumericalVector<double>()));
                    contravariantBaseVectors->insert(pair<Direction, NumericalVector<double>>(Three, NumericalVector<double>()));
                    break;
                default:
                    throw runtime_error("Invalid number of dimensions! You are getting into Einsteins field->");
            }*/

            covariantTensor = make_shared<NumericalMatrix<double>>(dimensions, dimensions);
            contravariantTensor = make_shared<NumericalMatrix<double>>(dimensions, dimensions);
            jacobian = make_shared<double>();
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

