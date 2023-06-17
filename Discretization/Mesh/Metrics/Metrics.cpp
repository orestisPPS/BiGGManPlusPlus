//
// Created by hal9000 on 4/12/23.
//

#include "Metrics.h"

namespace Discretization{
    
        Metrics::Metrics(Node* node, unsigned dimensions) {
            this->node = node;
            covariantBaseVectors = make_shared<map<Direction, vector<double>>>();
            contravariantBaseVectors = make_shared<map<Direction, vector<double>>>();
            covariantTensor = make_shared<Array<double>>(dimensions, dimensions);
            contravariantTensor = make_shared<Array<double>>(dimensions, dimensions);

/*            switch (dimensions) {
                case 1:
                    covariantBaseVectors->insert(pair<Direction, vector<double>>(One, vector<double>()));
                    contravariantBaseVectors->insert(pair<Direction, vector<double>>(One, vector<double>()));
                    break;
                case 2:
                    covariantBaseVectors->insert(pair<Direction, vector<double>>(One, vector<double>()));
                    covariantBaseVectors->insert(pair<Direction, vector<double>>(Two, vector<double>()));
                    
                    contravariantBaseVectors->insert(pair<Direction, vector<double>>(One, vector<double>()));
                    contravariantBaseVectors->insert(pair<Direction, vector<double>>(Two, vector<double>()));
                    break;
                case 3:
                    covariantBaseVectors->insert(pair<Direction, vector<double>>(One, vector<double>()));
                    covariantBaseVectors->insert(pair<Direction, vector<double>>(Two, vector<double>()));
                    covariantBaseVectors->insert(pair<Direction, vector<double>>(Three, vector<double>()));
                    
                    contravariantBaseVectors->insert(pair<Direction, vector<double>>(One, vector<double>()));
                    contravariantBaseVectors->insert(pair<Direction, vector<double>>(Two, vector<double>()));
                    contravariantBaseVectors->insert(pair<Direction, vector<double>>(Three, vector<double>()));
                    break;
                default:
                    throw runtime_error("Invalid number of dimensions! You are getting into Einsteins field->");
            }*/

            covariantTensor = make_shared<Array<double>>(dimensions, dimensions);
            contravariantTensor = make_shared<Array<double>>(dimensions, dimensions);
            jacobian = make_shared<double>();
        }
        
        void Metrics::calculateCovariantTensor() const {
            auto n = covariantBaseVectors->size();
            for (auto i = 0; i < n; i++) {
                for (auto j = 0; j < n; j++) {
                    auto gi = covariantBaseVectors->at(unsignedToSpatialDirection[i]);
                    auto gj = covariantBaseVectors->at(unsignedToSpatialDirection[j]);
                    auto gij = VectorOperations::dotProduct(gi, gj);
                    covariantTensor->at(i, j) = gij;
                }
            }
        }
        
        void Metrics::calculateContravariantTensor() const {
            auto n = contravariantBaseVectors->size();
            for (auto i = 0; i < n; i++) {
                for (auto j = 0; j < n; j++) {
                    auto gi = contravariantBaseVectors->at(unsignedToSpatialDirection[i]);
                    auto gj = contravariantBaseVectors->at(unsignedToSpatialDirection[j]);
                    auto gij = VectorOperations::dotProduct(gi, gj);
                    contravariantTensor->at(i, j) = gij;
                }
            }
        }
    
    } // Discretization

