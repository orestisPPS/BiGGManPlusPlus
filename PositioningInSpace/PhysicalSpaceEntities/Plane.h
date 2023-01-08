//
// Created by hal9000 on 1/8/23.
//

#ifndef UNTITLED_PLANE_H
#define UNTITLED_PLANE_H

#include "PhysicalSpaceEntity.h"

namespace PositioningInSpace {
        class Plane : PhysicalSpaceEntity {
        public:
            Plane(PhysicalSpaceEntities type);
            PhysicalSpaceEntities  type() override;
        private:
            PhysicalSpaceEntities _type;
            static bool _checkInput(PhysicalSpaceEntities type);
        };
} // PositioningInSpace

#endif //UNTITLED_PLANE_H
