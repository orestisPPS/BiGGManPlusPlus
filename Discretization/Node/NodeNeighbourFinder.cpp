//
// Created by hal9000 on 1/22/23.
//

#include "NodeNeighbourFinder.h"

namespace Discretization {
    
    map<Position, unsigned> find_neighbors(unsigned id, unsigned nn1, unsigned nn2, unsigned nn3) {
        map<Position, unsigned> neighbours;
        auto maxId = nn1 * nn2 * nn3 - 1;
        auto k = id / (nn1 * nn2);
        auto j = (id - k * nn1 * nn2) / nn1;
        auto i = id - k * nn1 * nn2 - j * nn1;
        
        // Central slice of a cube that includes the central node
        if (id + nn1 - 1 <= maxId) neighbours[TopLeft] = id + nn1 - 1;
        if (id + nn1 <= maxId) neighbours[Top] = id + nn1;
        if (id + nn1 + 1 <= maxId) neighbours[TopRight] = id + nn1 + 1;
        if (id - 1 <= maxId) neighbours[Left] = id - 1;
        if (id + 1 <= maxId) neighbours[Right] = id + 1;
        if (id - nn1 - 1 <= maxId) neighbours[BottomLeft] = id - nn1 - 1;
        if (id - nn1 <= maxId) neighbours[Bottom] = id - nn1;
        if (id - nn1 + 1 <= maxId) neighbours[BottomRight] = id - nn1 + 1;
        
        // Front slice of a cube
        if (k > 0) {
            if (id - nn2 - nn1 - 1 <= maxId) neighbours[FrontTopLeft] = id - nn2 - nn1 - 1;
            if (id - nn2 - nn1 <= maxId) neighbours[FrontTop] = id - nn2 - nn1;
            if (id - nn1 - nn2 + 1 <= maxId) neighbours[FrontTopRight] = id - nn1 - nn2 + 1;
            if (id - 2 * nn1 - nn2 - 1 <= maxId) neighbours[FrontLeft] = id - 2 * nn1 - nn2 - 1;
            if (id - 2 * nn1 - nn2 <= maxId) neighbours[Front] = id - 2 * nn1 - nn2;
            if (id - 2 * nn1 - nn2 + 1 <= maxId) neighbours[FrontRight] = id - 2 * nn1 - nn2 + 1;
            if (id - 3 * nn1 - nn2 - 1 <= maxId) neighbours[FrontBottomLeft] = id - 3 * nn1 - nn2 - 1;
            if (id - 3 * nn1 - nn2 <= maxId) neighbours[FrontBottom] = id - 3 * nn1 - nn2;
            if (id - 3 * nn1 - nn2 + 1 <= maxId) neighbours[FrontBottomRight] = id - 3 * nn1 - nn2 + 1;
        }
        
        // Back slice of a cube
        if (k < nn3 - 1) {
            if (id + 3 * nn1 + nn2 - 1 <= maxId) neighbours[BackTopLeft] = id + 3 * nn1 + nn2 - 1;
            if (id + 3 * nn1 + nn2 <= maxId) neighbours[BackTop] = id + 3 * nn1 + nn2;
            if (id + 3 * nn1 + nn2 + 1 <= maxId) neighbours[BackTopRight] = id + 3 * nn1 + nn2 + 1;
            if (id + 2 * nn1 + nn2 - 1 <= maxId) neighbours[BackLeft] = id + 2 * nn1 + nn2 - 1;
            if (id + 2 * nn1 + nn2 <= maxId) neighbours[Back] = id + 2 * nn1 + nn2;
            if (id + 2 * nn1 + nn2 + 1 <= maxId) neighbours[BackRight] = id + 2 * nn1 + nn2 + 1;
            if (id + nn1 + nn2 - 1 <= maxId) neighbours[BackBottomLeft] = id + nn1 + nn2 - 1;
            if (id + nn1 + nn2 <= maxId) neighbours[BackBottom] = id + nn1 + nn2;
            if (id + nn1 + nn2 + 1 <= maxId) neighbours[BackBottomRight] = id + nn1 + nn2 + 1;
        }

        
        //Remove the neighbours that are outside the mesh
        //TODO: remove later if all the above conditions are correct
        for (auto it = neighbours.begin(); it != neighbours.end(); it++) {
            if (it->second < 0 || it->second >= maxId) {
                neighbours.erase(it);
            }
        }

    } // Discretization
} // Discretization