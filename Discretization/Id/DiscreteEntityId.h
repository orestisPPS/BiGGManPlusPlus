//
// Created by hal9000 on 11/28/22.
//

#ifndef UNTITLED_DISCRETEENTITYID_H
#define UNTITLED_DISCRETEENTITYID_H

#pragma once

namespace Discretization
{
    class DiscreteEntityId {
    public:

        explicit DiscreteEntityId();

        ~DiscreteEntityId();

        //The global value of the entity
        unsigned *global;
    };
}


#endif //UNTITLED_DISCRETEENTITYID_H
