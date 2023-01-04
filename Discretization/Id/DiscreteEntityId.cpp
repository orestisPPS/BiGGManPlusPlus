//
// Created by hal9000 on 11/28/22.
//

#include "DiscreteEntityId.h"
#include <iostream>
#include <limits>

namespace Discretization
{
    DiscreteEntityId::DiscreteEntityId() {
        global = new unsigned(std::numeric_limits<int>::quiet_NaN());
        boundary = new unsigned(std::numeric_limits<int>::quiet_NaN());
        internal = new unsigned(std::numeric_limits<int>::quiet_NaN());
    }

    DiscreteEntityId::~DiscreteEntityId() {
        delete global;
        global = nullptr;
        delete boundary;
        boundary = nullptr;
        delete internal;
        internal = nullptr;
    }
}
