#ifndef  UNTITLED_NODALCOORDINATES_H
#define UNTITLED_NODALCOORDINATES_H

#include <map>
#include "../../PositioningInSpace/DirectionsPositions.h"

namespace Discretization {

    enum CoordinateType {
        Natural, Parametric, Template
    };
    
/**
* @brief The class that holds the nodal coordinates for each node.
* 
* This class holds a map of the nodal coordinates with the key being the type of coordinates (Natural, Parametric, Template).
* It also provides functions to add, remove and replace coordinate vectors in the map.
*/
    class NodalCoordinates {
    public:
        /**
        * @brief The constructor for the NodalCoordinates class.
        * 
        * Initializes the map of position vectors.
        */
        NodalCoordinates();
        
        ~NodalCoordinates();

        /**
        * @brief The overloaded operator that returns the natural coordinate at the specified index.
        * 
        * @param i The index of the natural coordinate.
        * @return const double& The natural coordinate at the specified index.
        */
        const double &operator()(unsigned i) const;

        /**
        * @brief The overloaded operator that returns the coordinate of the specified type at the specified index.
        * 
        * @param type The type of coordinate.
        * @param i The index of the coordinate.
        * @return const double& The coordinate of the specified type at the specified index.
        */
        const double &operator()(CoordinateType type, unsigned i) const;

        /**
        * @brief Adds a new coordinate vector to the map of position vectors with the specified type.
        * 
        * @param positionVector The pointer to the vector of coordinates to be added.
        * @param type The type of coordinates.
        */
        void addPositionVector(vector<double> *positionVector, CoordinateType type);

        /**
        * @brief Adds a new natural coordinate vector to the map of position vectors.
        * 
        * @param positionVector The pointer to the vector of natural coordinates to be added.
        */
        void addPositionVector(vector<double> *positionVector);

        /**
        * @brief Adds a new coordinate vector to the map of position vectors with the specified type.
        * 
        * @param type The type of coordinates.
        */
        void addPositionVector(CoordinateType type);

        /**
        * @brief Replaces the coordinate vector of the specified type with the new coordinate vector.
        * 
        * @param positionVector The pointer to the new vector of coordinates.
        * @param type The type of coordinates.
        */
        void setPositionVector(vector<double> *positionVector, CoordinateType type);

        /**
        * @brief Replaces the natural coordinate vector with the new natural coordinate vector.
        * 
        * @param positionVector The pointer to the new vector of natural coordinates.
        */
        void setPositionVector(vector<double> *positionVector);

        /**
        * @brief Removes the coordinate vector of the specified type from the map of position vectors.
        * 
        * @param type The type of coordinates to be removed.
        */
        void removePositionVector(CoordinateType type);

        /**
        * @brief Returns the vector of natural coordinates.
        * 
        * @return const vector<double>& The vector of natural coordinates.
        */
        const vector<double> &positionVector();

        /**
        * @brief Returns a pointer to the vector of natural coordinates.
        * 
        * @return vector<double>* A pointer to the vector of natural coordinates.
        */
        vector<double> *positionVectorPtr();
        
        /**
         * @brief Returns the vector of coordinates of the specified type.
         *
         * @param type 
         * @return const vector<double>& The vector of coordinates of the specified type. 
         */
        const vector<double>& positionVector(CoordinateType type);
        
        /**
        * @brief Returns a pointer to the vector of coordinates of the specified type.
        * 
        * @param type  The type of coordinates.
        * @return vector<double>* A pointer to the vector of coordinates of the specified type.
        */
        vector<double> *positionVectorPtr(CoordinateType type);
        
    private:
        map<CoordinateType, vector<double> *> *_positionVectors; ///< The map of position vectors with the key being the type of coordinates.
    };
} // Discretization

#endif //UNTITLED_NODALCOORDINATES_H