#ifndef  UNTITLED_NODALCOORDINATES_H
#define UNTITLED_NODALCOORDINATES_H

#include <map>
#include <stdexcept>
#include <memory>
#include "../../PositioningInSpace/DirectionsPositions.h"

namespace Discretization {

/**
 * \enum CoordinateType
 * \brief Enumeration for defining the type of nodal coordinates.
 */
    enum CoordinateType {
        Natural,      ///< Natural coordinates.
        Parametric,   ///< Parametric coordinates.
        Template      ///< Template coordinates.
    };
    
    /**
     * \class NodalCoordinates
     * \brief This class represents a the collection of the coordinates of the node.
     *
     * The nodal coordinates can be of different types, i.e., Natural, Parametric, or Template.
     * This class provides methods for getting, setting, and removing these coordinates.
     */
    class NodalCoordinates {
    public:

        /**
         * \brief Default constructor.
         * Initializes the map of position vectors.
         */
        NodalCoordinates();

        /**
         * \brief Destructor.
         * Clears the position vectors.
         */
        ~NodalCoordinates();

        /**
        * \brief Overloaded assignment operator for the NodalCoordinates class.
        * \param other The other NodalCoordinates object to assign from.
        * \return Reference to the updated NodalCoordinates object.
        */
        NodalCoordinates& operator=(const NodalCoordinates& other);
        
        /**
         * \brief Access operator for natural coordinates.
         * \param i Index of the coordinate.
         * \param type Type of the coordinate (default is Natural).
         * \return The ith natural coordinate.
         */
        const double &operator()(unsigned i, CoordinateType type = Natural) const;

        /**
         * \brief Gets the position vector of the specified type.
         * \param type Type of the coordinate (default is Natural).
         * \return Shared pointer to the position vector of the specified type.
         */
        const shared_ptr<NumericalVector<double>> &getPositionVector(CoordinateType type = Natural);

        /**
         * \brief Gets a copy of the elements of the position vector in 3D for the specified type.
         * If the size of the original vector is less than 3, fills the rest with zeros.
         * \param type Type of the coordinate (default is Natural).
         * \return A NumericalVector containing the 3D coordinates.
         */
        NumericalVector<double> getPositionVector3D(CoordinateType type = Natural);

        /**
         * \brief Sets the position vector for the specified type. 
         * If it exists, it is overwritten. If it does not exist, it is added.
         * \param positionVector The new position vector.
         * \param type Type of the coordinate (default is Natural).
         */
        void setPositionVector(shared_ptr<NumericalVector<double>> positionVector, CoordinateType type = Natural);

        /**
         * \brief Removes the position vector of the specified type.
         * \param type Type of the coordinate.
         */
        void removePositionVector(CoordinateType type = Natural);

    private:
        unique_ptr<map<CoordinateType, shared_ptr<NumericalVector<double>>>> _positionVectors; ///< The map of position vectors with the key being the type of coordinates.
    };

} // Discretization
#endif //UNTITLED_NODALCOORDINATES_H