//
// Created by hal9000 on 8/16/23.
//

#ifndef UNTITLED_NUMERICALVECTOR_H
#define UNTITLED_NUMERICALVECTOR_H

#include <vector>
#include <stdexcept>
#include <memory>
#include <thread>
#include <type_traits>
#include <valarray>
#include <random>
#include "../../../ThreadingOperations/ThreadingOperations.h"
using namespace LinearAlgebra;
using namespace std;

namespace LinearAlgebra {
    enum VectorNormType2 {
        // L1 (Manhattan / Taxicab) norm
        // The sum of the absolute values of the vectors' components.
        // For a vector x with n components, the L1 norm is denoted as ||x||1 and defined as:
        // ||x||1 = |x₁| + |x₂| + ... + |xₙ|
        L12,

        // L2 (Euclidean) norm
        // The square root of the sum of the squares of the vectors' components.
        // For a vector x with n components, the L2 norm is denoted as ||x||2 and defined as:
        // ||x||2 = √(x₁² + x₂² + ... + xₙ²)
        L22,

        // L∞ (Chebyshev) norm
        // The maximum absolute value of the vectors' components.
        // For a vector x with n components, the L∞ norm is denoted as ||x||∞ and defined as:
        // ||x||∞ = max(|x₁|, |x₂|, ..., |xₙ|)
        LInf2,


        // Lp norm
        // The pth root of the sum of the pth powers of the vectors' components.    
        // For a vector x with n components, the Lp norm is denoted as ||x||p and defined as:
        // ||x||p = (|x₁|^p + |x₂|^p + ... + |xₙ|^p)^(1/p)
        Lp2,

/*        //Frobenius (Euclidean for matrices) norm
        // Defined only for Array class. 
        // The square root of the sum of the squares of the matrices' components.
        // For a matrix A with m rows and n columns, the Frobenius norm is denoted as ||A||F and defined as:
        // ||A||F = √(A₁₁² + A₁₂² + ... + Aₘₙ²)
        Frobenius*/
    };
    
    template<typename T>
    class NumericalVector {

    public:
        
        /**
        * @brief Constructs a new NumericalVector object.
        * 
        * @param size Size of the numerical vector.
        * @param initialValue Default value for vector elements.
        * @param parallelizationMethod Parallelization method to be used for vector operations.
        */
        explicit NumericalVector(unsigned int size, T initialValue = 0, unsigned availableThreads = 1){
            
            static_assert(std::is_arithmetic<T>::value, "Template type T must be an arithmetic type (integral or floating-point)");
            _values = make_shared<vector<T>> (size, initialValue);
            _availableThreads = availableThreads;
        }

        /**
        * @brief Constructs a new NumericalVector object.
        * @param values Initial values for the vector.
        * @param parallelizationMethod Parallelization method to be used for vector operations.
        */
        NumericalVector(std::initializer_list<T> values, unsigned availableThreads = 1) {
            
            static_assert(std::is_arithmetic<T>::value, "Template type T must be an arithmetic type (integral or floating-point)");
            _values = make_shared<vector<T>>(values);
            _availableThreads = availableThreads;
        }



        /**
        * @brief Destructor for NumericalVector.
        * 
        * Cleans up and deallocates the vector.
        */
        ~NumericalVector() {

        }

        //=================================================================================================================//
        //=================================================== Operators ====================================================//
        //=================================================================================================================//

        /**
        * @brief Copy constructor.
        * 
        * @param other The source object to be copied from.
        */
        NumericalVector(const NumericalVector<T> &other){
            static_assert(std::is_arithmetic<T>::value, "Template type T must be an arithmetic type (integral or floating-point)");
            _availableThreads = other._availableThreads;
            _deepCopy(other);
        }
        
        
        /**
        * @brief Copy constructor.
        * @param other Constant reference to the source object to be copied from.
        */
        explicit NumericalVector(const std::shared_ptr<NumericalVector<T>> &other) {
            static_assert(std::is_arithmetic<T>::value, "Template type T must be an arithmetic type (integral or floating-point)");
            _deepCopy(other);
            _availableThreads = other->_availableThreads;
        }
        
        /**
         * @brief Construct a new Numerical Vector object from a unique pointer.
         * @param other Unique pointer to the source object to be copied from.
         */
        explicit NumericalVector(const std::unique_ptr<NumericalVector<T>> &other) {
            static_assert(std::is_arithmetic<T>::value, "Template type T must be an arithmetic type (integral or floating-point)");
            _deepCopy(other);
            _availableThreads = other->_availableThreads;
        }
        
        /**
         * @brief Construct a new Numerical Vector object from a raw pointer.
         * @param other Pointer to the source object to be copied from.
         */
        explicit NumericalVector(const NumericalVector<T> *other) {
            static_assert(std::is_arithmetic<T>::value, "Template type T must be an arithmetic type (integral or floating-point)");
            _deepCopy(other);
            _availableThreads = other->_availableThreads;
        }

        /**
        * @brief Overloaded assignment operator.
        * 
        * @param other The source object to be copied from.
        * @return Reference to the current object.
        */
        template<typename InputType>
        NumericalVector &operator=(const InputType &other) {
            
            if (this != &other) {
                _deepCopy(other);
                _availableThreads = other.getAvailableThreads();
            }
            return *this;
        }

        /**
        * @brief Overloaded equality operator.
        * 
        * @param other The object to be compared with.
        * @return true if the objects have the same data, false otherwise.
        */
        template<typename InputType>
        bool operator==(const InputType &other) const {
            _checkInputType(other);
            // Using the dereference trait to obtain the pointer to the source data
            T *otherData = dereference_trait<InputType>::dereference(other);
            unsigned int otherSize = dereference_trait<InputType>::size(other);
            return _areElementsEqual(otherData, otherSize);
        }

        /**
        * @brief Overloaded inequality operator.
        * 
        * @param other The object to be compared with.
        * @return true if the objects do not have the same data, false otherwise.
        */
        template<typename InputType>
        bool operator!=(const InputType &other) const {
            return !(*this == other);
        }

        /**
        * @brief Accesses the element at the specified index.
        * 
        * @param index Index of the element to access.
        * @return T& Reference to the element at the specified index.
        */
        T &operator[](unsigned int index) {
            if (index >= _values->size()) {
                throw out_of_range("Index out of range.");
            }
            return (*_values)[index];
        }


        /**
        * @brief Accesses the element at the specified index (const version).
        * 
        * @param index Index of the element to access.
        * @return const T& Constant reference to the element at the specified index.
        */
        const T &operator[](unsigned int index) const {
            if (index >= _values->size()) {
                throw out_of_range("Index out of range.");
            }
            return (*_values)[index];
        }

        /**
        * @brief Accesses the element at the specified index with bounds checking.
        * 
        * Throws std::out_of_range exception if index is out of bounds.
        * 
        * @param index Index of the element to access.
        * @return T& Reference to the element at the specified index.
        */
        T &at(unsigned int index) {
            if (index >= _values->size()) {
                throw out_of_range("Index out of range.");
            }
            return (*_values)[index];
        }

        /**
        * @brief Accesses the element at the specified index with bounds checking (const version).
        * 
        * Throws std::out_of_range exception if index is out of bounds.
        * 
        * @param index Index of the element to access.
        * @return const T& Constant reference to the element at the specified index.
        */
        const T &at(unsigned int index) const {
            if (index >= _values->size()) {
                throw out_of_range("Index out of range.");
            }
            return (*_values)[index];
        }
        
        //=================================================================================================================//
        //=================================================== Iterators ===================================================//
        //=================================================================================================================//
        
        /**
        * \brief Type definition for iterator.
        */
        using iterator = typename std::vector<T>::iterator;

        /**
         * \brief Type definition for constant iterator.
         */
        using const_iterator = typename std::vector<T>::const_iterator;

        /**
         * \brief Returns an iterator to the beginning of the vector.
         * \return An iterator to the beginning of the vector.
         */
        iterator begin() {
            return _values->begin();
        }

        /**
         * \brief Returns an iterator to the end of the vector.
         * \return An iterator to the end of the vector.
         */
        iterator end() {
            return _values->end();
        }

        /**
         * \brief Returns a constant iterator to the beginning of the vector.
         * \return A constant iterator to the beginning of the vector.
         */
        const_iterator begin() const {
            return _values->begin();
        }

        /**
         * \brief Returns a constant iterator to the end of the vector.
         * \return A constant iterator to the end of the vector.
         */
        const_iterator end() const {
            return _values->end();
        }

        /**
         * \brief Returns a constant iterator to the beginning of the vector.
         * \return A constant iterator to the beginning of the vector.
         */
        const_iterator cbegin() const {
            return _values->cbegin();
        }

        /**
         * \brief Returns a constant iterator to the end of the vector.
         * \return A constant iterator to the end of the vector.
         */
        const_iterator cend() const {
            return _values->cend();
        }

        /**
         * \brief Provides a range of iterators for a specific subrange of the vector.
         * \param start The starting index of the range.
         * \param end The ending index of the range.
         * \return A pair of iterators representing the beginning and end of the specified subrange.
         */
        std::pair<iterator, iterator> range(unsigned start, unsigned end) {
            if (start >= _values->size() || end > _values->size() || start > end) {
                throw std::out_of_range("Invalid range specified.");
            }
            return {_values->begin() + start, _values->begin() + end};
        }

        /**
         * \brief Provides a range of constant iterators for a specific subrange of the vector.
         * \param start The starting index of the range.
         * \param end The ending index of the range.
         * \return A pair of constant iterators representing the beginning and end of the specified subrange.
         */
        std::pair<const_iterator, const_iterator> range(unsigned start, unsigned end) const {
            if (start >= _values->size() || end > _values->size() || start > end) {
                throw std::out_of_range("Invalid range specified.");
            }
            return {_values->begin() + start, _values->begin() + end};
        }
        
        
        
        //=================================================================================================================//
        //=================================================== Utility ====================================================//
        //=================================================================================================================//

        /**
        * @brief Returns the size of the vector.
        * 
        * @return unsigned int Size of the vector.
        */
        unsigned int size() const {
            return _values->size();
        }
        
        
        /**
         * @brief Checks if the vector is empty.
         * @return true if the vector is empty, false otherwise.
         */
        bool empty() const {
            return _values->empty();
        }
        
        /**
        * @brief Fills the vector with the specified value.
        * @param value The value to fill the vector with.
        */
        void fill(T value) {
            auto fillJob = [&](unsigned int start, unsigned int end) {
                for (unsigned int i = start; i < end; i++) {
                    (*_values)[i] = value;
                }
            };
            _threading.executeParallelJob(fillJob, _values->size(), _availableThreads);
        }

        /**
        * \brief Fills the vector with random values between the specified minimum and maximum.
        * 
        * This function uses the Mersenne Twister random number generator to generate 
        * random values. The generator is seeded with a device-dependent random number 
        * to ensure different sequences in different program runs.
        * 
        * \tparam T The data type of the vector components (e.g., double, float).
        * 
        * \param min The minimum value for the random numbers.
        * \param max The maximum value for the random numbers.
        */
        void fillRandom(T min, T max) {
            //Mersenne Twister generator seeded with a device-dependent random number
            // Static to retain state across calls
            static std::mt19937 generator(std::random_device{}());
            std::uniform_real_distribution<T> distribution(min, max);

            for (auto &component: *_values) {
                component = distribution(generator);
            }
        }
        
        /**
         * @brief Resizes the vector. If the new size is smaller than the current size, the vector is truncated.
         * If the new size is larger than the current size, the vector is extended and the new elements are initialized
         * with the specified value.
         * @param newSize The new size of the vector.
         * @param initialValue The value to initialize the new elements with.
         */
        void resize(unsigned int newSize, T initialValue = 0) {
            _values->resize(newSize, initialValue);
        }
        
        /**
        * @brief Clears the vector calling the clear() method of the underlying std::vector<T>.
        */
        void clear() {
            _values->clear();
        }


        /**
         * @brief Returns a pointer to the underlying data.
         * @return T* Pointer to the underlying data.
         */
        T *getDataPointer() const {
            return _values->data();
        }
        
        /**
         * @brief Returns a shared pointer to the underlying data std vector<T>.
         * @return shared_ptr<vector<T>> Shared pointer to the underlying data.
         */
        shared_ptr<vector<T>>& getData() {
            return _values;
        }

        //=================================================================================================================//
        //=================================================== Threading   =================================================//
        //=================================================================================================================//

        /**
        * @brief Returns the number of threads used for vector operations.
        * @return unsigned int The number of threads used for vector operations.
        */
        unsigned getAvailableThreads() const{
            return _availableThreads;
        }

        /**
        * @brief Sets the number of threads to be used for vector operations.
        * @param availableThreads The number of threads to be used for vector operations.
         * @throws runtime_error If the number of threads is 0.
         * @throws runtime_error If the number of threads exceeds the number of available CPU cores.
        */
        void setAvailableThreads(unsigned availableThreads){
            if (availableThreads == 0){
                throw runtime_error("Number of threads must be greater than 0.");
            }
            if (availableThreads > thread::hardware_concurrency()){
                throw runtime_error("Number of threads cannot exceed the number of available CPU cores.");
            }
            _availableThreads = availableThreads;
        }


        //=================================================================================================================//
        //============================================== Numerical Operations =============================================//
        //=================================================================================================================//
        
        /**
        * @brief Computes the sum of the elements of the NumericalVector.
        * 
        * This method employs parallel processing to compute the sum and then aggregates the results.
        * 
        * @return T The sum of the elements of the NumericalVector.
        */
        T sum() {
            auto sumElementsJob = [&](unsigned start, unsigned end) -> T {
                T sum = 0;
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    sum += (*_values)[i];
                }
                return sum;
            };

            return ThreadingOperations<T>::executeParallelJobWithReduction(sumElementsJob, _values->size(), _availableThreads);
            
        }

        /**
         * @brief Computes the magnitude (Euclidean norm) of the NumericalVector.
         * 
         * This method employs parallel processing to compute the sum of squares and then aggregates the results.
         * The square root of the aggregated sum is taken to get the magnitude.
         * 
         * @return double The magnitude of the NumericalVector.
         */
        double magnitude() {
            auto magnitudeJob = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    sum += (*_values)[i] * (*_values)[i];
                }
                return sum;
            };

            return sqrt(_threading.executeParallelJobWithReductionForDoubles(magnitudeJob, _values->size(), _availableThreads));
        }

        /**
        * @brief Normalizes this vector.
        * 
        * Normalization of a vector refers to the process of dividing each component of the vector by the vector magnitude. 
        * After normalization, the magnitude of the vector will be 1.
        */
        void normalize() {
            double vectorMagnitude = magnitude();
            if (vectorMagnitude == 0) {
                throw std::runtime_error("Cannot normalize a vector with magnitude 0.");
            }
            scale(1.0 / vectorMagnitude);
        }

        /**
        * @brief Normalizes this vector to a custom magnitude.
        * @param customMagnitude The magnitude to normalize to.
        * 
        * Normalization of a vector refers to the process of dividing each component of the vector by the vector magnitude.
        * After normalization, the magnitude of the vector will be equal to the custom magnitude.
        */
        void normalizeTo(double customMagnitude) {
            if (customMagnitude == 0) {
                throw std::runtime_error("Cannot normalize a vector with magnitude 0.");
            }
            scale(1.0 / customMagnitude);
        }

        /**
        * @brief Calculates the Euclidean distance between this vector and the input vector.
        * @param inputVector Input vector to calculate the distance to.
        * @return The distance between the two vectors.
        * @throws invalid_argument If the input vectors are of different sizes.
        * 
        * The distance between two vectors v and w is defined as the magnitude of their difference.
        * distance(v, w) = magnitude(v - w)
        */
        template<typename InputType>
        double distance(const InputType &inputVector) {
            
            _checkInputType(inputVector);
            if (size() != dereference_trait<InputType>::size(inputVector)) {
                throw invalid_argument("Vectors must be of the same size.");
            }

            //T* inputVectorData = dereference_trait<InputType>::data(inputVector);
            const T *inputVectorData = dereference_trait<InputType>::dereference(inputVector);


            auto distanceThreadJob = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    sum += ((*_values)[i] - inputVectorData[i]) * ((*_values)[i] - inputVectorData[i]);
                }
                return sum;
            };

            double distanceSquared = _threading.executeParallelJobWithReductionForDoubles(distanceThreadJob, size(), _availableThreads);
            return sqrt(distanceSquared);
        }


        /**
        * @brief Calculates the angle between this vector and the input vector.
        * @param inputVector Input vector to calculate the distance to.
        * @return The angle between the two vectors.
        * @throws invalid_argument If the input vectors are of different sizes.
        * 
        * The angle θ between two vectors v and w is given by the formula:
        * cos(θ) = (dot(v, w)) / (magnitude(v) * magnitude(w))
        */
        template<typename InputType>
        auto angle(const InputType &inputVector) {

            _checkInputType(inputVector);
            if (size() != dereference_trait<InputType>::size(inputVector)) {
                throw invalid_argument("Vectors must be of the same size.");
            }
            const T *inputVectorData = dereference_trait<InputType>::dereference(inputVector);


            double dot = dotProduct(inputVector);


            double thisMagnitude = magnitude();

            auto inputMagnitudeThreadJob = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    sum += inputVectorData[i] * inputVectorData[i];
                }
                return sum;
            };

            double inputMagnitudeSquared = _threading.executeParallelJobWithReductionForDoubles(inputMagnitudeThreadJob, size(), _availableThreads);
            double inputMagnitude = sqrt(inputMagnitudeSquared);

            if (thisMagnitude == 0 || inputMagnitude == 0) {
                throw std::runtime_error("Cannot compute angle between vectors with magnitude 0.");
            }
            double cosValue = dot / (thisMagnitude * inputMagnitude);
            cosValue = std::max(-1.0, std::min(1.0, cosValue));
            return acos(cosValue);

        }

        
        /**
         * @brief Projects one vector onto another and stores the result in a third vector.
         *
         * This function computes the projection of the provided input vector onto the current instance vector
         * and stores the result in the result vector.
         *
         * The projection of vector B (toBeProjected) onto vector A (current instance) is computed using the formula:
         * proj_A B = (B . A) / (A . A) * A
         * This result is stored in the result vector.
         *
         * @tparam InputType1 The type of the input vector that needs to be projected. Can be a raw pointer,
         *         a shared pointer, a unique pointer, or a direct object of NumericalVector.
         * @tparam InputType2 The type of the result vector where the projection result will be stored. Can be
         *         a raw pointer, a shared pointer, a unique pointer, or a direct object of NumericalVector.
         * @param toBeProjected The vector that is to be projected onto the current instance.
         * @param result The vector where the projection result will be stored.
         * @throws std::invalid_argument If any of the vectors are not of the same size.
         * @throws std::runtime_error If the magnitude of the input vector to be projected is zero.
         */
        template<typename InputType1, typename InputType2>
        void project(const InputType1 &toBeProjected, InputType2 &result){

            _checkInputType(toBeProjected);
            _checkInputType(result);
            if (size() != dereference_trait<InputType1>::size(toBeProjected)) {
                throw invalid_argument("Vectors must be of the same size.");
            }
            if (size() != dereference_trait<InputType2>::size(result)) {
                throw invalid_argument("Vectors must be of the same size.");
            }
            const T *inputVectorData1 = dereference_trait<InputType1>::dereference(toBeProjected);
            T *resultData = dereference_trait<InputType2>::dereference(result);

            double dot = dotProduct(toBeProjected);

            auto inputMagnitudeThreadJob = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end && i < size(); ++i) {
                    sum += inputVectorData1[i] * inputVectorData1[i];
                }
                return sum;
            };

            double inputMagnitudeSquared = _threading.executeParallelJobWithReductionForDoubles(inputMagnitudeThreadJob, size(), _availableThreads);
            double inputMagnitude = sqrt(inputMagnitudeSquared);

            if (inputMagnitude == 0) {
                throw std::runtime_error("Cannot project on a vector with magnitude 0.");
            }
            double scalar = dot / inputMagnitudeSquared;
            
            auto projectionThreadJob = [&](unsigned start, unsigned end) {
                for (unsigned i = start; i < end && i < size(); ++i) {
                    resultData[i] = inputVectorData1[i] * scalar;
                }
            };
            _threading.executeParallelJob(projectionThreadJob, size(), _availableThreads);
        }

        /**
         * @brief Projects a given vector onto the current instance vector.
         *
         * This function computes the projection of the provided input vector onto the current instance vector
         * and updates the current instance with the result.
         *
         * The projection of vector B (toBeProjected) onto vector A (current instance) is computed using the formula:
         * proj_A B = (B . A) / (A . A) * A
         * This result updates the current instance vector.
         *
         * @tparam InputType The type of the vector that needs to be projected. Can be a raw pointer,
         *         a shared pointer, a unique pointer, or a direct object of NumericalVector.
         * @param toBeProjected The vector that is to be projected onto the current instance.
         * @throws std::invalid_argument If the vectors are not of the same size.
         * @throws std::runtime_error If the magnitude of the vector to be projected is zero.
         */
        template<typename InputType>
        void projectOntoThis(const InputType &toBeProjected) {
            _checkInputType(toBeProjected);
            if (size() != dereference_trait<InputType>::size(toBeProjected)) {
                throw invalid_argument("Vectors must be of the same size.");
            }
            const T *inputVectorData1 = dereference_trait<InputType>::dereference(toBeProjected);

            double dot = dotProduct(toBeProjected);

            auto inputMagnitudeThreadJob = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end && i < size(); ++i) {
                    sum += inputVectorData1[i] * inputVectorData1[i];
                }
                return sum;
            };

            double inputMagnitudeSquared = _threading.executeParallelJobWithReductionForDoubles(inputMagnitudeThreadJob, size(), _availableThreads);
            double inputMagnitude = sqrt(inputMagnitudeSquared);

            if (inputMagnitude == 0) {
                throw std::runtime_error("Cannot project on a vector with magnitude 0.");
            }
            double scalar = dot / inputMagnitudeSquared;

            auto projectionThreadJob = [&](unsigned start, unsigned end) {
                for (unsigned i = start; i < end && i < size(); ++i) {
                    (*_values)[i] = inputVectorData1[i] * scalar;
                }
            };
            _threading.executeParallelJob(projectionThreadJob, size(), _availableThreads);
        }

        /**
         * @brief Applies the Householder transformation to a given vector.
         * 
         * Householder transformation is a method used to zero out certain components of a vector.
         * It reflects a given vector about a plane or hyperplane. This function applies the
         * transformation  to the current instance vector and stores the result in the result vector.
         * 
         * Mathematically, the Householder matrix is defined as:
         *    H = I - 2 * (v * v^T) / (v^T * v)
         * where `v` is the reflection vector. This function uses this concept to reflect the
         * the current instance vector and stores the result in the result vector.
         * 
         * @tparam InputType The type of the result vector. The function supports various types
         *                   like raw pointers, shared pointers, and the NumericalVector itself.
         * 
         * @param targetVector 
         * 
         * @throw invalid_argument if the size of the input vector and the current vector don't match.
         */
        template<typename InputType>
        void houseHolderTransformation() {
            
            //Lambda function to calculate the sign of a number
            auto sign = [](double x) -> int {
                return x >= 0.0 ? 1 : -1;
            };
            
            //Calculate the norm of the current vector
            double norm = normL2();
            double alpha = -sign((*_values)[0]) * norm;
            (*_values)[0] -= alpha;
            normalize();
        }

        /**
         * @brief Applies the Householder transformation to a given vector.
         * 
         * Householder transformation is a method used to zero out certain components of a vector.
         * It reflects a given vector about a plane or hyperplane. This function applies the
         * transformation  to the current instance vector and stores the result in the result vector.
         * 
         * Mathematically, the Householder matrix is defined as:
         *    H = I - 2 * (v * v^T) / (v^T * v)
         * where `v` is the reflection vector. This function uses this concept to reflect the current instance 
         * of the vector.
         */
        template<typename InputType>
        void houseHolderTransformationIntoThis(){
            auto sign = [](double x) -> int {
                return x >= 0.0 ? 1 : -1;
            };
            double norm = normL2();
            double alpha = -sign((*_values)[0]) * norm;
            auto range = std::make_pair(1, size());
            (*_values)[0] -= alpha;
        }
        

        /**
        * @brief Calculates the sum of all the elements of this vector.
        * @return The sum of all components of the vector.
        * 
        * Given a vector v = [v1, v2, ..., vn], the sum is calculated as:
        * average(v) = (v1 + v2 + ... + vn) / n
        */
        double average() {
            return sum() / _values->size();
        }


        /**
        @brief Calculates the variance of the elements of this vector.
        @return The variance of the vector.
        * 
        * The variance of a vector is a measure of how spread out its values are. It is calculated as the average of the
        * squared differences between each element and the mean of the vector.
        * Given a vector x with n elements, the variance is calculated as:
        * variance(x) = (1/n) * (sum of (x[i] - mean(x))^2 for i=1 to n)
        */
        double variance() {
            double averageOfElements = average();

            auto varianceJob = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    double diff = (*_values)[i] - averageOfElements;
                    sum += diff * diff;
                }
                return sum;
            };

            return _threading.executeParallelJobWithReductionForDoubles(varianceJob, size(), _availableThreads) / size();
        }

        /**
        * @brief Calculates the standard deviation of the elements of this vector.
        * @return The standard deviation of the vector.
        * 
        * The standard deviation of a vector is the square root of its variance. It is a measure of how spread out its values
        * are, but it is expressed in the same units as the original data.
        */
        double standardDeviation() {
            return sqrt(variance());
        }

        /**
        * @brief Calculates the covariance between this vector and the input vector.
        * @param inputVector Input vector to calculate the covariance to.
        * @return The covariance between the two vectors.
        * 
        * The covariance between two vectors is a measure of how they vary together. If the covariance is positive, the values of
        * one vector tend to be high when the values of the other vector are high, and low when the values of the other vector are
        * low. If the covariance is negative, the values of one vector tend to be high when the values of the other vector are low,
        * and vice versa. A covariance of zero means that the vectors are uncorrelated, i.e., their values do not vary together.
        * Given two vectors X = [x1, x2, ..., xn] and Y = [y1, y2, ..., yn], their covariance is calculated as:
        * cov(X, Y) = 1/n * sum((xi - mean(X)) * (yi - mean(Y))), where mean(X) and mean(Y) are the means of X and Y, respectively.
        * If the two vectors have the same length, the formula simplifies to:
        * cov(X, Y) = 1/n * dot(X - mean(X), Y - mean(Y)), where dot() is the dot product.
        */
        template<typename InputType>
        double covariance(const InputType &inputVector) {

            _checkInputType(inputVector);
            if (size() != dereference_trait<InputType>::size(inputVector)) {
                throw invalid_argument("Vectors must be of the same size.");
            }

            const T *inputVectorData = dereference_trait<InputType>::dereference(inputVector);

            double averageOfThis = average();

            auto averageOfInputJob = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end; ++i) {
                    sum += inputVectorData[i];
                }
                return sum;
            };

            double averageOfInput = _threading.executeParallelJobWithReductionForDoubles(averageOfInputJob, size(), _availableThreads) / _values->size();

            auto covarianceJob = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    sum += ((*_values)[i] - averageOfThis) * (inputVectorData[i] - averageOfInput);
                }
                return sum;
            };

            return _threading.executeParallelJobWithReductionForDoubles(covarianceJob, size(), _availableThreads) / size();
        }

        /**
        * @brief Calculates the correlation between this vector and the input vector.
        * @param inputVector Input vector to calculate the correlation to.
        * @return The correlation between the two vectors.
        * 
        * The correlation coefficient between two vectors is a measure of how strong the linear relationship is between them. It
        * ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no correlation.
        * Given two vectors X = [x1, x2, ..., xn] and Y = [y1, y2, ..., yn], their correlation coefficient is calculated as:
        * cor(X, Y) = cov(X, Y) / (sd(X) * sd(Y)), where cov(X, Y) is the covariance between X and Y, and sd(X) and sd(Y) are the
        * standard deviations of X and Y, respectively.
        */
        template<typename InputType>
        double correlation(const InputType &inputVector) {

            _checkInputType(inputVector);
            if (size() != dereference_trait<InputType>::size(inputVector)) {
                throw invalid_argument("Vectors must be of the same size.");
            }

            T *inputVectorData = dereference_trait<InputType>::dereference(inputVector);
            double covarianceOfVectors = covariance(inputVector);
            double stdDevOfThis = standardDeviation();

            // Calculate mean of inputVector
            auto meanOfInputJob = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end; ++i) {
                    sum += inputVectorData[i];
                }
                return sum;
            };
            double meanOfInput = _threading.executeParallelJobWithReductionForDoubles(meanOfInputJob, size(), _availableThreads) / _values->size();

            // Calculate variance of inputVector
            auto varianceOfInputJob = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end; ++i) {
                    double diff = inputVectorData[i] - meanOfInput;
                    sum += diff * diff;
                }
                return sum;
            };
            double varianceOfInput = _threading.executeParallelJobWithReductionForDoubles(varianceOfInputJob, size(), _availableThreads) / _values->size();
            double stdDevOfInput = sqrt(varianceOfInput);

            if (stdDevOfThis == 0 || stdDevOfInput == 0) {
                throw std::runtime_error("Cannot compute correlation between vectors with standard deviation 0.");
            }

            return covarianceOfVectors / (stdDevOfThis * stdDevOfInput);
        }
        
        double norm(VectorNormType2 normType, double p = 1){
            switch (normType){
                case VectorNormType2::L12:
                    return normL1();
                case VectorNormType2::L22:
                    return normL2();
                case VectorNormType2::LInf2:
                    return normLInf();
                case VectorNormType2::Lp2:
                    return normLp(p);
                default:
                    throw std::runtime_error("Invalid norm type.");
            }
        }
        
        double normL1() {
            auto normL1Job = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    sum += abs((*_values)[i]);
                }
                return sum;
            };

            return _threading.executeParallelJobWithReductionForDoubles(normL1Job, size(), _availableThreads);
        }

        double normL2() {
            auto normL2Job = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    sum += (*_values)[i] * (*_values)[i];
                }
                return sum;
            };

            return sqrt(_threading.executeParallelJobWithReductionForDoubles(normL2Job, size(), _availableThreads));
        }

        double normLInf() {
            auto normLInfJob = [&](unsigned start, unsigned end) -> double {
                double maxVal = 0;
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    double absValue = abs((*_values)[i]);
                    if (absValue > maxVal) {
                        maxVal = absValue;
                    }
                }
                return maxVal;
            };
            auto reductionResults = _threading.executeParallelJobWithIncompleteReduction(normLInfJob, size(), _availableThreads);
            return *std::max_element(reductionResults.begin(), reductionResults.end());
        }

        double normLp(double p) {
            auto normLpJob = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    sum += pow(abs((*_values)[i]), p);
                }
                return sum;
            };

            return pow(_threading.executeParallelJobWithReductionForDoubles(normLpJob, size()), _availableThreads);
        }

        //=================================================================================================================//
        //================================================== Vector Operations ============================================//
        //=================================================================================================================//
        /**
        * \brief Calculates the dot product of 2 vectors.
        * The dot product of two vectors is defined as the sum of the products of their corresponding components.
        * Given two vectors v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], their dot product is calculated as:
        * dot(v, w) = v1w1 + v2w2 + ... + vn*wn
        * Geometrically, the dot product of two vectors gives the cosine of the angle between them multiplied by the magnitudes
        * of the vectors. If the dot product is zero, it means the vectors are orthogonal (perpendicular) to each other.
        * The subtraction is performed in parallel across multiple threads.
        *
        * \tparam T The data type of the vectors (e.g., double, float).
        * 
        * \param vector The input vector.
         * @return T The dot product of the two vectors.
        */
        template<typename InputType>
        T dotProduct(const InputType &vector, unsigned userDefinedThreads = 0) {
            
            _checkInputType(vector);
            if (size() != dereference_trait<InputType>::size(vector)) {
                throw invalid_argument("Vectors must be of the same size.");
            }

            T *otherData = dereference_trait<InputType>::dereference(vector);

            auto dotProductJob = [&](unsigned start, unsigned end) -> T {
                T localDotProduct = 0;
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    localDotProduct += (*_values)[i] * otherData[i];
                }
                return localDotProduct;
            };

            unsigned availableThreads = (userDefinedThreads > 0) ? userDefinedThreads : _availableThreads;
            return _threading.executeParallelJobWithReduction(dotProductJob, _values->size(), availableThreads);
        }
        

        /**
        * \brief Performs element-wise addition of two scaled vectors.
        * Given two vectors v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], and scalar factors a and b, their addition is:
        * add(v, w) = [a*v1+b*w1, a*v2+b*w2, ..., a*vn+b*wn]. The addition is performed in parallel across multiple threads.
        * 
        * \param inputVector The input vector to add.
        * \param result The result vector after addition.
        * \param scaleThis Scaling factor for the current vector (default is 1).
        * \param scaleInput Scaling factor for the input vector (default is 1).
        */
        template<typename InputType1, typename InputType2>
        void add(const InputType1 &inputVector, InputType2 &result, T scaleThis = 1, T scaleInput = 1, unsigned userDefinedThreads = 0) {
            
            _checkInputType(inputVector);
            _checkInputType(result);
            if (size() != dereference_trait<InputType1>::size(inputVector)) {
                throw invalid_argument("Vectors must be of the same size.");
            }
            if (size() != dereference_trait<InputType2>::size(result)) {
                throw invalid_argument("Vectors must be of the same size.");
            }

            const T *otherData = dereference_trait<InputType1>::dereference(inputVector);
            T *resultData = dereference_trait<InputType2>::dereference(result);

            auto addJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    resultData[i] = scaleThis * (*_values)[i] + scaleInput * otherData[i];
                }
            };
            unsigned availableThreads = (userDefinedThreads > 0) ? userDefinedThreads : _availableThreads;
            _threading.executeParallelJob(addJob, _values->size(), availableThreads);
        }

        /**
        * \brief Performs element-wise addition of a scaled vector into the current vector.
        * Given vector v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], and scalar factors a and b, the updated vector is:
        * v = [a*v1+b*w1, a*v2+b*w2, ..., a*vn+b*wn]. The addition is performed in parallel across multiple threads.
        * 
        * \param inputVector The input vector to add.
        * \param scaleThis Scaling factor for the current vector (default is 1).
        * \param scaleInput Scaling factor for the input vector (default is 1).
        */
        template<typename InputType>
        void addIntoThis(const InputType &inputVector, T scaleThis = 1, T scaleInput = 1, unsigned userDefinedThreads = 0) {
            _checkInputType(inputVector);
            if (size() != dereference_trait<InputType>::size(inputVector)) {
                throw invalid_argument("Vectors must be of the same size.");
            }

            const T *otherData = dereference_trait<InputType>::dereference(inputVector);
            auto addJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    (*_values)[i] = scaleThis * (*_values)[i] + scaleInput * otherData[i];
                }
            };
            unsigned availableThreads = (userDefinedThreads > 0) ? userDefinedThreads : _availableThreads;
            _threading.executeParallelJob(addJob, _values->size(), availableThreads);
        }

        /**
        * \brief Performs element-wise subtraction of two scaled vectors.
        * Given two vectors v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], and scalar factors a and b, their subtraction is:
        * subtract(v, w) = [a*v1-b*w1, a*v2-b*w2, ..., a*vn-b*wn]. The subtraction is performed in parallel across multiple threads.
        * 
        * \param inputVector The input vector to subtract.
        * \param result The result vector after subtraction.
        * \param scaleThis Scaling factor for the current vector (default is 1).
        * \param scaleInput Scaling factor for the input vector (default is 1).
        */
        template<typename InputType1, typename InputType2>
        void subtract(const InputType1 &inputVector, InputType2 &result, T scaleThis = 1, T scaleInput = 1, unsigned userDefinedThreads = 0) {
            
            _checkInputType(inputVector);
            _checkInputType(result);
            if (size() != dereference_trait<InputType1>::size(inputVector)) {
                throw invalid_argument("Vectors must be of the same size.");
            }
            if (size() != dereference_trait<InputType2>::size(result)) {
                throw invalid_argument("Vectors must be of the same size.");
            }

            const T *otherData = dereference_trait<InputType1>::dereference(inputVector);
            T *resultData = dereference_trait<InputType2>::dereference(result);

            auto subtractJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    resultData[i] = scaleThis * (*_values)[i] - scaleInput * otherData[i];
                }
            };
            unsigned availableThreads = (userDefinedThreads > 0) ? userDefinedThreads : _availableThreads;
            _threading.executeParallelJob(subtractJob, _values->size(), availableThreads);
        }

        /**
        * \brief Performs element-wise subtraction of a scaled vector from the current vector.
        * Given vector v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], and scalar factors a and b, the updated vector is:
        * v = [a*v1-b*w1, a*v2-b*w2, ..., a*vn-b*wn]. The subtraction is performed in parallel across multiple threads.
        * 
        * \param inputVector The input vector to subtract.
        * \param scaleThis Scaling factor for the current vector (default is 1).
        * \param scaleInput Scaling factor for the input vector (default is 1).
        */
        template<typename InputType>
        void subtractIntoThis(const InputType &inputVector, T scaleThis = 1, T scaleInput = 1, unsigned userDefinedThreads = 0) {
            _checkInputType(inputVector);
            if (size() != dereference_trait<InputType>::size(inputVector)) {
                throw invalid_argument("Vectors must be of the same size.");
            }

            const T *otherData = dereference_trait<InputType>::dereference(inputVector);
            auto subtractJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    (*_values)[i] = scaleThis * (*_values)[i] - scaleInput * otherData[i];
                }
            };
            unsigned availableThreads = (userDefinedThreads > 0) ? userDefinedThreads : _availableThreads;
            _threading.executeParallelJob(subtractJob, _values->size(), availableThreads);
        }
        
        /**
         * @brief Scales the current vector by a given scalar.
         * @param scalar The scalar to scale the vector by.
         */
        void scale(T scalar, unsigned userDefinedThreads = 0) {
            auto scaleJob = [&](unsigned start, unsigned end) -> void {
                for (auto &value: *_values) {
                    value *= scalar;
                }
            };
            unsigned availableThreads = (userDefinedThreads > 0) ? userDefinedThreads : _availableThreads;
            _threading.executeParallelJob(scaleJob, _values->size(), _availableThreads);
        }


    protected:
        shared_ptr<vector<T>> _values; ///< The underlying data.
        
        ThreadingOperations<T> _threading; ///< Threading operations used for parallelization.
        
        unsigned _availableThreads; ///< The number of threads available for operations on this vector.
        

        /**
        * @brief Performs a deep copy from the source to the current object.
        * 
        * This method uses the dereference_trait to handle various types of sources 
        * such as raw pointers, unique pointers, shared pointers, and direct objects.
        * 
        * @param source The source object to be copied from.
        */
        template<typename InputType>
        void _deepCopy(const InputType &source) {

            if (size() != dereference_trait<InputType>::size(source)) {
                throw std::invalid_argument("Source vector must be the same size as the destination vector.");
            }

            const T *sourceData = dereference_trait<InputType>::dereference(source);

            auto deepCopyThreadJob = [&](unsigned start, unsigned end) {
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    (*_values)[i] = sourceData[i];
                }
            };
            _threading.executeParallelJob(deepCopyThreadJob, _values->size(), _availableThreads);
        }

        /**
        * @brief Checks if the elements of the current object are equal to those of the provided source.
        * 
        * This method uses parallelization to perform the comparison and then reduces the results to determine 
        * if all elements are equal.
        * 
        * @param source The source object to be compared with.
        * @return true if all elements are equal, false otherwise.
        */
        bool _areElementsEqual(T * const &source, size_t size) const {

            if (_values->size() != size) {
                throw std::invalid_argument("Source vector must be the same size as the destination vector.");
            }

            auto compareElementsJob = [&](unsigned start, unsigned end) -> bool {
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    if ((*_values)[i] != source[i]) {
                        return false;
                    }
                }
                return true;
            };

            return _threading.executeParallelJobWithReduction(compareElementsJob, _values->size(), _availableThreads);
        }



        //=================================================================================================================//
        //============================================ Dereference Traits =================================================//
        //=================================================================================================================//

        /**
        * \brief Trait to standardize dereferencing of various types.
        *
        * This trait provides a unified way to dereference types such as raw pointers,
        * unique pointers, shared pointers, and direct objects.
        */
        template<typename U>
        struct dereference_trait;

        /**
        * \brief Base trait for raw pointers and direct objects.
        *
        * This trait provides a unified way to dereference types like NumericalVector and 
        * raw pointers to NumericalVector.
        */
        template<typename U>
        struct dereference_trait_base {
            /**
            * \brief Dereferences the source.
            * \param source A pointer to the source object.
            * \return A pointer to the data of the source.
            */
            static U *dereference(U *source) {
                static_assert(std::is_arithmetic<U>::value, "Template type T must be an arithmetic type (integral or floating-point)");

                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->getDataPointer();
            }

            /**
            * \brief Fetches the size of the source.
            * \param source A pointer to the source object.
            * \return The size of the source.
            */
            static unsigned size(U *source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->size();
            }
        };

        /// Specialization for NumericalVector<U>.
        template<typename U>
        struct dereference_trait<NumericalVector<U>> {

            /**
            * \brief Dereferences the source.
            * \param source A smart pointer to the source object.
            * \return A pointer to the data of the source.
            */
            static U* dereference(const NumericalVector<U> &source) {
                static_assert(std::is_arithmetic<U>::value, "Template type T must be an arithmetic type (integral or floating-point)");
                return source.getDataPointer();
            }
            
            /**
            * \brief Fetches the size of the source.
            * \param source A smart pointer to the source object.
            * \return The size of the source.
            */
            static unsigned size(const NumericalVector<U> &source) {
                return source.size();
            }
        };


        /// Specialization for raw pointer to NumericalVector<U>.
        template<typename U>
        struct dereference_trait<NumericalVector<U> *> : public dereference_trait_base<NumericalVector<U>> {
        };

        /**
        * \brief Base trait for smart pointers.
        *
        * This trait provides a unified way to dereference types like std::unique_ptr and 
        * std::shared_ptr.
        */
        template<template<typename, typename...> class PtrType, typename U>
        struct dereference_trait_pointer_base {
            /**
            * \brief Dereferences the source.
            * \param source A smart pointer to the source object.
            * \return A pointer to the data of the source.
            */
            static U *dereference(const PtrType<NumericalVector<U>> &source) {
                static_assert(std::is_arithmetic<U>::value, "Template type T must be an arithmetic type (integral or floating-point)");
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->getDataPointer();
            }

            static ThreadingOperations<U> &threadingOperations(const PtrType<NumericalVector<U>> &source) {
                return source->_threading;
            }
            
            /**
            * \brief Fetches the size of the source.
            * \param source A smart pointer to the source object.
            * \return The size of the source.
            */
            static unsigned size(const PtrType<NumericalVector<U>> &source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->size();
            }
        };
        
        /// Specialization for std::unique_ptr<NumericalVector<U>>.
        template<typename U>
        struct dereference_trait<std::unique_ptr<NumericalVector<U>>>
                : public dereference_trait_pointer_base<std::unique_ptr, U> {
        };

        /// Specialization for std::shared_ptr<NumericalVector<U>>.
        template<typename U>
        struct dereference_trait<std::shared_ptr<NumericalVector<U>>>
                : public dereference_trait_pointer_base<std::shared_ptr, U> {
        };
        
        template<typename InputType>
        void _checkInputType(InputType const &input) const {
            static_assert(std::is_same<InputType, NumericalVector<T>>::value
                          || std::is_same<InputType, std::shared_ptr<NumericalVector<T>>>::value
                          || std::is_same<InputType, std::unique_ptr<NumericalVector<T>>>::value
                          || std::is_same<InputType, NumericalVector<T>*>::value,
                          "Input must be a NumericalVector, its pointer, or its smart pointers.");
        }
    };
    
}

#endif //UNTITLED_NUMERICALVECTOR_H
