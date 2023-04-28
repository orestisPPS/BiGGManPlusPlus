//
// Created by hal9000 on 4/12/23.
//

#include "Metrics.h"

namespace Discretization{
    
        Metrics::Metrics(Node* node, unsigned dimensions) {
            this->node = node;
            covariantBaseVectors = new map<Direction, vector<double>>();
            contravariantBaseVectors = new map<Direction, vector<double>>();
            covariantTensor = new Array<double>(dimensions, dimensions);
            contravariantTensor = new Array<double>(dimensions, dimensions);

            switch (dimensions) {
                case 1:
                    covariantBaseVectors->insert(pair<Direction, vector<double>>(One, vector<double>(dimensions)));
                    contravariantBaseVectors->insert(pair<Direction, vector<double>>(One, vector<double>(dimensions)));
                    break;
                case 2:
                    covariantBaseVectors->insert(pair<Direction, vector<double>>(One, vector<double>(dimensions)));
                    covariantBaseVectors->insert(pair<Direction, vector<double>>(Two, vector<double>(dimensions)));
                    
                    contravariantBaseVectors->insert(pair<Direction, vector<double>>(One, vector<double>(dimensions)));
                    contravariantBaseVectors->insert(pair<Direction, vector<double>>(Two, vector<double>(dimensions)));
                    break;
                case 3:
                    covariantBaseVectors->insert(pair<Direction, vector<double>>(One, vector<double>(dimensions)));
                    covariantBaseVectors->insert(pair<Direction, vector<double>>(Two, vector<double>(dimensions)));
                    covariantBaseVectors->insert(pair<Direction, vector<double>>(Three, vector<double>(dimensions)));
                    
                    contravariantBaseVectors->insert(pair<Direction, vector<double>>(One, vector<double>(dimensions)));
                    contravariantBaseVectors->insert(pair<Direction, vector<double>>(Two, vector<double>(dimensions)));
                    contravariantBaseVectors->insert(pair<Direction, vector<double>>(Three, vector<double>(dimensions)));
                    break;
                default:
                    throw runtime_error("Invalid number of dimensions! You are getting into Einsteins field->");
            }

            covariantTensor = new Array<double>(dimensions, dimensions);
            contravariantTensor = new Array<double>(dimensions, dimensions);
            jacobian = new double;
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
        
        Metrics::~Metrics() {
            delete covariantBaseVectors;
            delete contravariantBaseVectors;
            delete covariantTensor;
            delete contravariantTensor;
            delete jacobian;
        }
    
    } // Discretization

