#ifndef UNTITLED_DOMAINBOUNDARYCONDITIONS_H
#define UNTITLED_DOMAINBOUNDARYCONDITIONS_H

#include <list>
#include <exception>
#include <stdexcept>
#include "BoundaryCondition.h"
#include "../../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"

using namespace PositioningInSpace;
using namespace DegreesOfFreedom;
using namespace std;

/**
 * @namespace MathematicalEntities
 * @brief Defines a set of boundary conditions for the domain.
 */
namespace MathematicalEntities {

    /**
     * @class DomainBoundaryConditions
     * @brief Handles the boundary conditions of a domain.
     */
    class DomainBoundaryConditions {
    public:

        explicit DomainBoundaryConditions();
        
        
        /**
         * @brief Constructs the domain boundary conditions with a specific map of positions and boundary conditions.
         *
         * @param bcAtPosition Pointer to a map of positions and their corresponding boundary conditions.
         */
        explicit DomainBoundaryConditions(shared_ptr<map<Position, shared_ptr<BoundaryCondition>>> bcAtPosition);
        


        /**
         * @brief Constructs the domain boundary conditions with a specific map of positions and nodal boundary conditions.
         *
         * @param nodalBcAtPosition Pointer to a map of positions and their corresponding nodal boundary conditions.
         */
        explicit DomainBoundaryConditions(shared_ptr<map <Position, shared_ptr<map<unsigned int*, shared_ptr<BoundaryCondition>>>>> nodalBcAtPosition);

        /**
         * @brief Gets the boundary condition at a specified position and nodeID.
         *
         * This method will return the boundary condition at the given position if it exists, 
         * otherwise it will throw an exception. If _bcAtPosition is not null, it will try to find 
         * the boundary condition in it. If _bcAtPosition is null but _nodalBcAtPosition is not null, 
         * it will try to find the boundary condition in _nodalBcAtPosition. If both are null, 
         * it will throw a runtime_error.
         *
         * @param boundaryPosition The position where the boundary condition is requested.
         * @param nodeID The ID of the node at the position.
         * @return Pointer to the boundary condition at the given position and nodeID.
         * @throws std::out_of_range If boundaryPosition or nodeID is not found in the respective map.
         * @throws std::runtime_error If no boundary conditions are found.
         */
        shared_ptr<BoundaryCondition> getBoundaryConditionAtPosition(Position boundaryPosition, unsigned* nodeID);
        
        void setBoundaryCondition(Position boundaryPosition, BoundaryConditionType bcType, DOFType dofType, double fixedValue);
        void setBoundaryCondition(std::initializer_list<Position> boundaryPositions, BoundaryConditionType bcType, DOFType dofType, double fixedValue);
         

    private:
        shared_ptr<map<Position, shared_ptr<map<unsigned int*, shared_ptr<BoundaryCondition>>>>> _nodalBcAtPosition;  ///< Holds the boundary conditions for each position and node ID.
        shared_ptr<map<Position, shared_ptr<BoundaryCondition>>> _bcAtPosition;  ///< Holds the boundary conditions for each position.
    };

} // MathematicalEntities

#endif //UNTITLED_DOMAINBOUNDARYCONDITIONS_H
