//
// Created by hal9000 on 4/12/23.
//

#include "Metrics.h"

namespace Discretization{
    
        Metrics::Metrics(Node* node, short unsigned dimensions) {
            this->node = node;
            covariantBaseVectors = new map<Direction, vector<double>>();
            contravariantBaseVectors = new map<Direction, vector<double>>();

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
        
        Metrics::~Metrics() {
            delete covariantTensor;
            delete contravariantTensor;
            delete jacobian;
        }
    
    } // Discretization

