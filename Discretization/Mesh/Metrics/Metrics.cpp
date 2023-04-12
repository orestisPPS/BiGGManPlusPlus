//
// Created by hal9000 on 4/12/23.
//

#include "Metrics.h"

namespace Discretization{
    
        Metrics::Metrics(Node* node, unsigned dimensions) {
            this->node = node;
            
            covariantBaseVectors = new map<Direction, vector<double>>();
            covariantBaseVectors->insert(pair<Direction, vector<double>>(One, vector<double>(dimensions)));
            covariantBaseVectors->insert(pair<Direction, vector<double>>(Two, vector<double>(dimensions)));
            
            contravariantBaseVectors = new map<Direction, vector<double>>();
            contravariantBaseVectors->insert(pair<Direction, vector<double>>(One, vector<double>(dimensions)));
            contravariantBaseVectors->insert(pair<Direction, vector<double>>(Two, vector<double>(dimensions)));
            
            covariantTensor = new Array<double>(dimensions, dimensions);
            contravariantTensor = new Array<double>(dimensions, dimensions);
            jacobian = new double;
        }
        
        Metrics::~Metrics() {
            delete covariantBaseVectors;
            delete contravariantBaseVectors;
            delete covariantTensor;
            delete contravariantTensor;
            delete jacobian;
        }
    
    } // Discretization

