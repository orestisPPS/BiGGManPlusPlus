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
#include "ParallelizationMethods.h"
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
        explicit NumericalVector(unsigned int size, T initialValue = 0,
                                 ParallelizationMethod parallelizationMethod = SingleThread) {
            _values = make_shared<vector<T>> (size, initialValue);
            _parallelizationMethod = parallelizationMethod;
        }

        /**
        * @brief Constructs a new NumericalVector object.
        * @param values Initial values for the vector.
        * @param parallelizationMethod Parallelization method to be used for vector operations.
        */
        NumericalVector(std::initializer_list<T> values, ParallelizationMethod parallelizationMethod = SingleThread) {
            _values = make_shared<vector<T>>(values);
            _parallelizationMethod = parallelizationMethod;
        }


        /**
        * @brief Destructor for NumericalVector.
        * 
        * Cleans up and deallocates the vector.
        */
        ~NumericalVector() {
            _values->clear();
            _values = nullptr;
        }

        //=================================================================================================================//
        //=================================================== Operators ====================================================//
        //=================================================================================================================//

        /**
        * @brief Copy constructor.
        * 
        * @param other The source object to be copied from.
        */
        NumericalVector(const NumericalVector<T> &other) {
            _deepCopy(other);
            _parallelizationMethod = other._parallelizationMethod;
        }
        
        
        /**
        * @brief Copy constructor.
        * @param other Constant reference to the source object to be copied from.
        */
        explicit NumericalVector(const std::shared_ptr<NumericalVector<T>> &other) {
            _deepCopy(other);
            _parallelizationMethod = other->_parallelizationMethod;
        }
        
        /**
         * @brief Construct a new Numerical Vector object from a unique pointer.
         * @param other Unique pointer to the source object to be copied from.
         */
        explicit NumericalVector(const std::unique_ptr<NumericalVector<T>> &other) {
            _deepCopy(other);
            _parallelizationMethod = other->_parallelizationMethod;
        }
        
        /**
         * @brief Construct a new Numerical Vector object from a raw pointer.
         * @param other Pointer to the source object to be copied from.
         */
        explicit NumericalVector(const NumericalVector<T> *other) {
            _deepCopy(other);
            _parallelizationMethod = other->_parallelizationMethod;
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
                _parallelizationMethod = dereference_trait<InputType>::parallelizationMethod(other);
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
            const T *otherData = dereference_trait<InputType>::dereference(other);
            return _areElementsEqual(otherData, other.size());
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
         * @brief Returns the parallelization method used for vector operations.
         * @return ParallelizationMethod The parallelization method used for vector operations.
         */
        ParallelizationMethod getParallelizationMethod() const {
            return _parallelizationMethod;
        }

        /**
         * @brief Checks if the vector is empty.
         * @return true if the vector is empty, false otherwise.
         */
        bool empty() const {
            return _values->empty();
        }

        /**
        * @brief Fill the vector with the specified value.
        * @param value The value to fill the vector with.
        */
        void fill(T value) {
            auto fillJob = [&](unsigned int start, unsigned int end) {
                for (unsigned int i = start; i < end; i++) {
                    (*_values)[i] = value;
                }
            };
            _executeParallelJob(fillJob);
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
         * @brief Returns a pointer to the underlying data.
         * @return T* Pointer to the underlying data.
         */
        T *data() const {
            return _values->data();
        }

        /**
        * @brief Computes the sum of the elements of the NumericalVector.
        * 
        * This method employs parallel processing to compute the sum and then aggregates the results.
        * 
        * @return T The sum of the elements of the NumericalVector.
        */
        T sum() const {
            auto sumElementsJob = [&](unsigned start, unsigned end) -> T {
                T sum = 0;
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    sum += (*_values)[i];
                }
                return sum;
            };

            if (_parallelizationMethod == SingleThread) {
                return _executeParallelJobWithReduction(_values->size(), sumElementsJob, 1);
            }

            if (_parallelizationMethod == MultiThread) {
                return _executeParallelJobWithReduction(_values->size(), sumElementsJob,
                                                        std::thread::hardware_concurrency());
            }
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

            return sqrt(_executeParallelJobWithReductionForDoubles(magnitudeJob));
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

            double distanceSquared = _executeParallelJobWithReductionForDoubles(distanceThreadJob);
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

            double inputMagnitudeSquared = _executeParallelJobWithReductionForDoubles(inputMagnitudeThreadJob);
            double inputMagnitude = sqrt(inputMagnitudeSquared);

            if (thisMagnitude == 0 || inputMagnitude == 0) {
                throw std::runtime_error("Cannot compute angle between vectors with magnitude 0.");
            }
            double cosValue = dot / (thisMagnitude * inputMagnitude);
            cosValue = std::max(-1.0, std::min(1.0, cosValue));
            return acos(cosValue);

        }

        /**
        * @brief Calculates the sum of all the elements of this vector.
        * @return The sum of all components of the vector.
        * 
        * Given a vector v = [v1, v2, ..., vn], the sum is calculated as:
        * sum(v) = v1 + v2 + ... + vn
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

            return _executeParallelJobWithReductionForDoubles(varianceJob) / _values->size();
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

            double averageOfInput = _executeParallelJobWithReductionForDoubles(averageOfInputJob) / _values->size();

            auto covarianceJob = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    sum += ((*_values)[i] - averageOfThis) * (inputVectorData[i] - averageOfInput);
                }
                return sum;
            };

            return _executeParallelJobWithReductionForDoubles(covarianceJob) / size();
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
            double meanOfInput = _executeParallelJobWithReductionForDoubles(meanOfInputJob) / _values->size();

            // Calculate variance of inputVector
            auto varianceOfInputJob = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end; ++i) {
                    double diff = inputVectorData[i] - meanOfInput;
                    sum += diff * diff;
                }
                return sum;
            };
            double varianceOfInput = _executeParallelJobWithReductionForDoubles(varianceOfInputJob) / _values->size();
            double stdDevOfInput = sqrt(varianceOfInput);

            if (stdDevOfThis == 0 || stdDevOfInput == 0) {
                throw std::runtime_error("Cannot compute correlation between vectors with standard deviation 0.");
            }

            return covarianceOfVectors / (stdDevOfThis * stdDevOfInput);
        }
        
        double normL1() {
            auto normL1Job = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    sum += abs((*_values)[i]);
                }
                return sum;
            };

            return _executeParallelJobWithReductionForDoubles(normL1Job);
        }

        double normL2() {
            auto normL2Job = [&](unsigned start, unsigned end) -> double {
                double sum = 0;
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    sum += (*_values)[i] * (*_values)[i];
                }
                return sum;
            };

            return sqrt(_executeParallelJobWithReductionForDoubles(normL2Job));
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
            auto reductionResults = vector<T>(
                    _parallelizationMethod == SingleThread ? 1 : std::thread::hardware_concurrency());
            if (_parallelizationMethod == SingleThread) {
                reductionResults = _executeParallelJobWithIncompleteReduction(_values->size(), normLInfJob, 1);
            } else if (_parallelizationMethod == MultiThread) {
                reductionResults = _executeParallelJobWithIncompleteReduction(_values->size(), normLInfJob,
                                                                              std::thread::hardware_concurrency());
            }
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

            return pow(_executeParallelJobWithReductionForDoubles(normLpJob), 1.0 / p);
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
        T dotProduct(const InputType &vector) {
            
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

            if (_parallelizationMethod == SingleThread) {
                return _executeParallelJobWithReduction(_values->size(), dotProductJob, 1);
            } else if (_parallelizationMethod == MultiThread) {
                return _executeParallelJobWithReduction(_values->size(), dotProductJob,
                                                        std::thread::hardware_concurrency());
            }
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
        void add(const InputType1 &inputVector, InputType2 &result, T scaleThis = 1, T scaleInput = 1) {
            
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
            _executeParallelJob(addJob);
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
        void addIntoThis(const InputType &inputVector, T scaleThis = 1, T scaleInput = 1) {
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
            _executeParallelJob(addJob);
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
        void subtract(const InputType1 &inputVector, InputType2 &result, T scaleThis = 1, T scaleInput = 1) {
            
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
            _executeParallelJob(subtractJob);
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
        void subtractIntoThis(const InputType &inputVector, T scaleThis = 1, T scaleInput = 1) {
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

            _executeParallelJob(subtractJob);
        }

        /**
        * \brief Scales the elements of the current vector by a given factor.
        * Given vector v = [v1, v2, ..., vn] and scalar factor a, the updated vector is:
        * v = [a*v1, a*v2, ..., a*vn]. The scaling is performed in parallel across multiple threads.
        * 
        * \param scale The scaling factor.
        */
        void scale(T scale) {
            auto scaleJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end && i < _values->size(); ++i) {
                    (*_values)[i] *= scale;
                }
            };
            _executeParallelJob(scaleJob);
        }

    private:
        shared_ptr<vector<T>> _values; ///< The underlying data.

        LinearAlgebra::ParallelizationMethod _parallelizationMethod; ///< Parallelization method used for matrix operations.


        /**
        * \brief Executes the provided task in parallel across multiple threads.
        * 
        * This method distributes the task across available CPU cores. Each thread operates on a distinct segment
        * of the data, ensuring parallel processing without race conditions.
        * 
        * \tparam ThreadJob A callable object type (function, lambda, functor).
        *
        * \param size The size of the data being processed.
        * \param task The callable object that describes the work each thread should execute.
         * \param availableThreads The number of threads available for processing.
        * \param cacheLineSize An optional parameter to adjust for system's cache line size (default is 64 bytes).
        */
        template<typename ThreadJob>
        static void
        _executeParallelJob(size_t size, ThreadJob task, unsigned availableThreads, unsigned cacheLineSize = 64) {
            unsigned doublesPerCacheLine = cacheLineSize / sizeof(double);
            unsigned int numThreads = std::min(availableThreads, static_cast<unsigned>(size));

            unsigned blockSize = (size + numThreads - 1) / numThreads;
            blockSize = (blockSize + doublesPerCacheLine - 1) / doublesPerCacheLine * doublesPerCacheLine;

            vector<thread> threads;
            for (unsigned int i = 0; i < numThreads; ++i) {
                unsigned start = i * blockSize;
                unsigned end = start + blockSize;
                if (start >= size) break;
                end = std::min(end, static_cast<unsigned>(size)); // Ensure 'end' doesn't exceed 'size'
                threads.push_back(thread(task, start, end));
            }

            for (auto &thread: threads) {
                thread.join();
            }
        }

        /**
        * \brief Executes the provided task in parallel across multiple threads with a reduction step.
        * 
        * This method distributes the task across available CPU cores. Each thread operates on a distinct segment
        * of the data and produces a local result. After all threads have completed their work, a reduction step
        * combines these local results into a single global result.
        * 
        * \tparam T The data type of the result (e.g., double, float).
        * \tparam ThreadJob A callable object type (function, lambda, functor).
        *
        * \param size The size of the data being processed.
        * \param task The callable object that describes the work each thread should execute and return a local result.
         * \param availableThreads The number of threads available for processing.
        * \param cacheLineSize An optional parameter to adjust for system's cache line size (default is 64 bytes).
        * 
        * \return The combined result after the reduction step.
        */
        template<typename ThreadJob>
        static T _executeParallelJobWithReduction(size_t size, ThreadJob task, unsigned availableThreads,
                                                  unsigned cacheLineSize = 64) {
            unsigned doublesPerCacheLine = cacheLineSize / sizeof(double);
            unsigned int numThreads = std::min(availableThreads, static_cast<unsigned>(size));

            unsigned blockSize = (size + numThreads - 1) / numThreads;
            blockSize = (blockSize + doublesPerCacheLine - 1) / doublesPerCacheLine * doublesPerCacheLine;

            vector<T> localResults(numThreads);
            vector<thread> threads;

            for (unsigned int i = 0; i < numThreads; ++i) {
                unsigned start = i * blockSize;
                unsigned end = start + blockSize;
                if (start >= size) break;
                end = std::min(end, static_cast<unsigned>(size)); // Ensure 'end' doesn't exceed 'size'
                threads.push_back(thread([&](unsigned start, unsigned end, unsigned idx) {
                    localResults[idx] = task(start, end);
                }, start, end, i));
            }

            for (auto &thread: threads) {
                thread.join();
            }

            T finalResult = 0;
            for (T val: localResults) {
                finalResult += val;
            }
            return finalResult;
        }

        /**
        * \brief Executes the provided task in parallel across multiple threads with an incomplete reduction step.
        * 
        * This method distributes the task across available CPU cores. Each thread operates on a distinct segment
        * of the data and produces a local result. After all threads have completed their work, a reduction step
        * combines these local results into a single global result.
        * 
        * \tparam T The data type of the result (e.g., double, float).
        * \tparam ThreadJob A callable object type (function, lambda, functor).
        *
        * \param size The size of the data being processed.
        * \param task The callable object that describes the work each thread should execute and return a local result.
         * \param availableThreads The number of threads available for processing.
        * \param cacheLineSize An optional parameter to adjust for system's cache line size (default is 64 bytes).
        * 
        * \return The result vector after the reduction step.
        */
        template<typename ThreadJob>
        static vector<T>
        _executeParallelJobWithIncompleteReduction(size_t size, ThreadJob task, unsigned availableThreads,
                                                   unsigned cacheLineSize = 64) {
            unsigned doublesPerCacheLine = cacheLineSize / sizeof(double);
            unsigned int numThreads = std::min(availableThreads, static_cast<unsigned>(size));

            unsigned blockSize = (size + numThreads - 1) / numThreads;
            blockSize = (blockSize + doublesPerCacheLine - 1) / doublesPerCacheLine * doublesPerCacheLine;

            vector<T> localResults(numThreads);
            vector<thread> threads;

            for (unsigned int i = 0; i < numThreads; ++i) {
                unsigned start = i * blockSize;
                unsigned end = start + blockSize;
                if (start >= size) break;
                end = std::min(end, static_cast<unsigned>(size)); // Ensure 'end' doesn't exceed 'size'
                threads.push_back(thread([&](unsigned start, unsigned end, unsigned idx) {
                    localResults[idx] = task(start, end);
                }, start, end, i));
            }

            for (auto &thread: threads) {
                thread.join();
            }
            return localResults;
        }

        template<typename ThreadJob>
        void _executeParallelJob(ThreadJob task) {
            if (_parallelizationMethod == SingleThread) {
                _executeParallelJob(_values->size(), task, 1);
            } else if (_parallelizationMethod == MultiThread) {
                _executeParallelJob(_values->size(), task, std::thread::hardware_concurrency());
            }
        }

        template<typename ThreadJob>
        double _executeParallelJobWithReductionForDoubles(ThreadJob task) {
            if (_parallelizationMethod == SingleThread) {
                return _executeParallelJobWithReduction(_values->size(), task, 1);
            } else if (_parallelizationMethod == MultiThread) {
                return _executeParallelJobWithReduction(_values->size(), task, std::thread::hardware_concurrency());
            }
        }


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
            _executeParallelJob(deepCopyThreadJob);
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
        bool _areElementsEqual(const T *&source, size_t size) {

            if (_values->size() != source->size()) {
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

            // Check elements in parallel and reduce the results
            if (_parallelizationMethod == SingleThread) {
                return _executeParallelJobWithReduction(_values->size(), compareElementsJob, 1);
            }

            if (_parallelizationMethod == MultiThread) {
                return _executeParallelJobWithReduction(_values->size(), compareElementsJob,
                                                        std::thread::hardware_concurrency());
            }
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
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->data();
            }


            /**
            * \brief Dereferences the source.
            * \param source A smart pointer to the source object.
            * \return The parallelization method of the source.
            */
            static ParallelizationMethod parallelizationMethod(const NumericalVector<U> &source) {
                return source->_parallelizationMethod;
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
                return source.data();
            }

            /**
            * \brief Dereferences the source.
            * \param source A smart pointer to the source object.
            * \return The parallelization method of the source.
            */
            static ParallelizationMethod parallelizationMethod(const NumericalVector<U> &source) {
                return source._parallelizationMethod;
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
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->data();
            }

            /**
            * \brief Dereferences the source.
            * \param source A smart pointer to the source object.
            * \return The parallelization method of the source.
            */
            static ParallelizationMethod parallelizationMethod(const PtrType<NumericalVector<U>> &source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->_parallelizationMethod;
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
        void _checkInputType(const InputType &input) {
            static_assert(std::is_same<InputType, NumericalVector<T>>::value
                          || std::is_same<InputType, std::shared_ptr<NumericalVector<T>>>::value
                          || std::is_same<InputType, std::unique_ptr<NumericalVector<T>>>::value
                          || std::is_same<InputType, NumericalVector<T>*>::value,
                          "Input must be a NumericalVector, its pointer, or its smart pointers.");
        }
    };
    
}

#endif //UNTITLED_NUMERICALVECTOR_H
