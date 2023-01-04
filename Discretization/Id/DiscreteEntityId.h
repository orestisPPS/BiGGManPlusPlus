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

        DiscreteEntityId();

        ~DiscreteEntityId();

        //The global id of the entity
        unsigned *global;

        //The boundary id of the entity. If it is not a boundary entity, this is set to NaN
        unsigned *boundary;

        //The internal id of the entity. If it is not an internal entity, this is set to NaN
        unsigned *internal;
    };
}


#endif //UNTITLED_DISCRETEENTITYID_H
