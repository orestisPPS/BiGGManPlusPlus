//
// Created by hal9000 on 10/22/23.
//

#ifndef UNTITLED_TIMEENTITY_H
#define UNTITLED_TIMEENTITY_H

namespace Discretization {

    class TimeEntity {
    public:
        TimeEntity(unsigned index, double currentTime, double currentTimeStep) :
                index(index), currentTime(currentTime), currentTimeStep(currentTimeStep) {}
        unsigned index;
        double currentTime;
        double currentTimeStep;
    };

} // Discretization

#endif //UNTITLED_TIMEENTITY_H
