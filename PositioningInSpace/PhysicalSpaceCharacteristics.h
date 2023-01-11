//
// Created by hal9000 on 1/5/23.
//

#ifndef UNTITLED_PHYSICALSPACECHARACTERISTICS_H
#define UNTITLED_PHYSICALSPACECHARACTERISTICS_H
#include <map>
#include <iostream>
#include "DirectionsPositions.h"
#include "CoordinateSystem.h"

namespace PositioningInSpace {

    class PhysicalSpaceCharacteristics {
        public:
            explicit PhysicalSpaceCharacteristics(std::map<PositioningInSpace::Direction, unsigned > &nodesPerDirection);
            unsigned _dimensions;
            PhysicalSpaceEntities physicalSpace;
            coordinateSystemType coordinateSystemType;
            
        private:
            static void CheckInput(std::map<PositioningInSpace::Direction, unsigned > &nodesPerDirection);
            void CalculateDimensions();
            void CalculatePhysicalSpace(std::map<PositioningInSpace::Direction, unsigned > &nodesPerDirection);
        };
    }; // PositioningInSpace

#endif //UNTITLED_PHYSICALSPACECHARACTERISTICS_H
