//
// Created by hal9000 on 4/14/23.
//

#ifndef UNTITLED_VECTOROPERATIONS_H
#define UNTITLED_VECTOROPERATIONS_H

#include <memory>
#include <vector>
#include <stdexcept>
#include <valarray>
using namespace std;

namespace LinearAlgebra {

    class VectorOperations {
        
    public:
        
        /**
        Calculates the dot product of two vectors.
        @param vector1 Constant reference to a shared pointer to the first vector.
        @param vector2 Constant reference to a shared pointer to the second vector.
        @return The dot product of the two vectors.
        @throws invalid_argument If the input vectors are of different sizes.
        
        The dot product of two vectors is defined as the sum of the products of their corresponding components.
        Given two vectors v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], their dot product is calculated as:
        dot(v, w) = v1w1 + v2w2 + ... + vn*wn
        Geometrically, the dot product of two vectors gives the cosine of the angle between them multiplied by the magnitudes
        of the vectors. If the dot product is zero, it means the vectors are orthogonal (perpendicular) to each other.
        */
        template<typename T>
        static T dotProduct(const shared_ptr<vector<T>> &vector1, const shared_ptr<vector<T>> &vector2) {
            auto result = 0.0;
            if (vector1->size() != vector2->size())
                throw invalid_argument("Vectors must have the same size");
            for (auto i = 0; i < vector1->size(); i++)
                result += vector1->at(i) * vector2->at(i);
            return result;
        }

        /**
        Calculates the dot product of two vectors.
        @param vector1 Constant reference to the first vector.
        @param vector2 Constant reference to  the second vector.
        @return The dot product of the two vectors.
        @throws invalid_argument If the input vectors are of different sizes.
        
        The dot product of two vectors is defined as the sum of the products of their corresponding components.
        Given two vectors v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], their dot product is calculated as:
        dot(v, w) = v1w1 + v2w2 + ... + vn*wn
        Geometrically, the dot product of two vectors gives the cosine of the angle between them multiplied by the magnitudes
        of the vectors. If the dot product is zero, it means the vectors are orthogonal (perpendicular) to each other.
        */
        template<typename T>
        static T dotProduct(const vector<T>& vector1, const vector<T>& vector2){
            auto result = 0.0;
            if (vector1.size() != vector2.size())
                throw invalid_argument("Vectors must have the same size");
            for (auto i = 0; i < vector1.size(); i++)
                result += vector1.at(i) * vector2.at(i);
            return result;
        }

        template<typename T>
        static T dotProductWithTranspose(const shared_ptr<vector<T>>& vector1){
            if (vector1->empty())
                throw invalid_argument("Vector must not be empty");
            auto result = 0.0;
            for (auto i = 0; i < vector1->size(); i++)
                result += vector1->at(i) * vector1->at(i);
            return result;
        }

        template<typename T>
        static T dotProductWithTranspose(const vector<T>& vector1){
            if (vector1.empty())
                throw invalid_argument("Vector must not be empty");
            auto result = 0.0;
            for (auto i = 0; i < vector1.size(); i++)
                result += vector1.at(i) * vector1.at(i);
            return result;
        }

        /**
        Calculates the cross product of two 3-dimensional vectors.
        @param vector1 Constant reference to a shared pointer to the first vector.
        @param vector2 Constant reference to a shared pointer to the second vector.
        @return The cross product of the two vectors.
        @throws invalid_argument If the input vectors are not 3-dimensional.
        
        The cross product of two 3-dimensional vectors is a vector that is perpendicular to both of them.
        Given two 3-dimensional vectors v = [v1, v2, v3] and w = [w1, w2, w3], their cross product is calculated as:
        cross(v, w) = [v2w3 - v3w2, v3w1 - v1w3, v1w2 - v2w1]
        */
        template<typename T>
        static vector<T> crossProduct(const shared_ptr<vector<T>>& vector1, const shared_ptr<vector<T>>& vector2){
            switch (vector1->size()) {
                case 2:
                    if (vector2->size() != 2)
                        throw invalid_argument("Vectors must have 2 dimensions");
                    return {vector1->at(0) * vector2->at(1) - vector1->at(1) * vector2->at(0)};
                case 3:
                    if (vector2->size() != 3)
                        throw invalid_argument("Vectors must have 3 dimensions");
                    return {vector1->at(1) * vector2->at(2) - vector1->at(2) * vector2->at(1),
                            vector1->at(2) * vector2->at(0) - vector1->at(0) * vector2->at(2),
                            vector1->at(0) * vector2->at(1) - vector1->at(1) * vector2->at(0)};
                default:
                    throw invalid_argument("Vectors must have 2 or 3 dimensions");
            }
        }

        /**
        Calculates the cross product of two 3-dimensional vectors.
        @param vector1 Constant reference to the first vector.
        @param vector2 Constant reference to  the second vector.
        @return The cross product of the two vectors.
        @throws invalid_argument If the input vectors are not 3-dimensional.
        
        The cross product of two 3-dimensional vectors is a vector that is perpendicular to both of them.
        Given two 3-dimensional vectors v = [v1, v2, v3] and w = [w1, w2, w3], their cross product is calculated as:
        cross(v, w) = [v2w3 - v3w2, v3w1 - v1w3, v1w2 - v2w1]
        */
        template<typename T>
        static vector<T> crossProduct(const vector<T>& vector1, const vector<T>& vector2){

            switch (vector1.size()) {
                case 2:
                    if (vector2.size() != 2)
                        throw invalid_argument("Vectors must have 2 dimensions");
                    return {vector1.at(0) * vector2.at(1) - vector1.at(1) * vector2.at(0)};
                case 3:
                    if (vector2.size() != 3)
                        throw invalid_argument("Vectors must have 3 dimensions");
                    return {vector1.at(1) * vector2.at(2) - vector1.at(2) * vector2.at(1),
                            vector1.at(2) * vector2.at(0) - vector1.at(0) * vector2.at(2),
                            vector1.at(0) * vector2.at(1) - vector1.at(1) * vector2.at(0)};
                default:
                    throw invalid_argument("Vectors must have 2 or 3 dimensions");
            }
        }

        /**
        * Scales each component of a vector by a scalar value.
        * @param vector Constant reference to a shared pointer to the input vector.
        * @param scalar The scaling factor to apply to each component of the vector.
        * 
        * Given a vector v = [v1, v2, ..., vn] and scalar s, the vector after scaling is:
        * scaled(v) = [s*v1, s*v2, ..., s*vn]
        */
        template<typename T, typename S>
        static void scale(const shared_ptr<vector<T>>& vector, S scalar){
            for (auto & i : *vector)
                i *= scalar;
        }

        /**
        * Overloaded method to scale each component of a vector by a scalar value.
        * @param vector Constant reference to the input vector.
        * @param scalar The scaling factor to apply to each component of the vector.
        * 
        * Given a vector v = [v1, v2, ..., vn] and scalar s, the vector after scaling is:
        * scaled(v) = [s*v1, s*v2, ..., s*vn]
        */
        template<typename T, typename S>
        static void scale(vector<T>& vector, S scalar){
            for (auto & i : vector)
                i *= scalar;
        }

        /**
        * Adds two vectors component-wise.
        * @param vector1 Constant reference to a shared pointer to the first vector.
        * @param vector2 Constant reference to a shared pointer to the second vector.
        * @return A vector that is the component-wise addition of the input vectors.
        * @throws invalid_argument If the input vectors are of different sizes.
        * 
        * Given two vectors v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], their addition is:
        * add(v, w) = [v1+w1, v2+w2, ..., vn+wn]
        */
        template<typename T>
        static vector<T> add(const shared_ptr<vector<T>>& vector1, const shared_ptr<vector<T>>& vector2){
            if (vector1->size() != vector2->size())
                throw invalid_argument("Vectors must have the same size");
            for (auto i = 0; i < vector1->size(); i++)
                vector1->at(i) += vector2->at(i);
        }

        /**
        * Overloaded method to add two vectors component-wise.
        * @param vector1 Constant reference to the first vector.
        * @param vector2 Constant reference to the second vector.
        * @return A vector that is the component-wise addition of the input vectors.
        * @throws invalid_argument If the input vectors are of different sizes.
        * 
        * Given two vectors v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], their addition is:
        * add(v, w) = [v1+w1, v2+w2, ..., vn+wn]
        */
        template<typename T>
        static vector<T> add(const vector<T>& vector1, const vector<T>& vector2){
            if (vector1.size() != vector2.size())
                throw invalid_argument("Vectors must have the same size");
            for (auto i = 0; i < vector1.size(); i++)
                vector1.at(i) += vector2.at(i);
        }

        /**
        * Subtracts the second vector from the first vector component-wise.
        * @param vector1 Constant reference to a shared pointer to the first vector.
        * @param vector2 Constant reference to a shared pointer to the second vector.
        * @return A vector that is the component-wise subtraction of the second vector from the first vector.
        * @throws invalid_argument If the input vectors are of different sizes.
        * 
        * Given two vectors v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], their subtraction is:
        * subtract(v, w) = [v1-w1, v2-w2, ..., vn-wn]
        */
        template<typename T>
        static vector<T> subtract(const shared_ptr<vector<T>>& vector1, const shared_ptr<vector<T>>& vector2){
            if (vector1->size() != vector2->size())
                throw invalid_argument("Vectors must have the same size");
            for (auto i = 0; i < vector1->size(); i++)
                vector1->at(i) -= vector2->at(i);
        }

        /**
        * Overloaded method to subtract the second vector from the first vector component-wise.
        * @param vector1 Constant reference to the first vector.
        * @param vector2 Constant reference to the second vector.
        * @return A vector that is the component-wise subtraction of the second vector from the first vector.
        * @throws invalid_argument If the input vectors are of different sizes.
        * 
        * Given two vectors v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], their subtraction is:
        * subtract(v, w) = [v1-w1, v2-w2, ..., vn-wn]
        */
        template<typename T>
        static vector<T> subtract(const vector<T>& vector1, const vector<T>& vector2){
            if (vector1.size() != vector2.size())
                throw invalid_argument("Vectors must have the same size");
            for (auto i = 0; i < vector1.size(); i++)
                vector1.at(i) -= vector2.at(i);
        }

        

        /**
        * Calculates the magnitude (or length) of a vector.
        * @param vector Constant reference to a shared pointer to the vector.
        * @return The magnitude of the vector.
        * 
        * The magnitude of a vector v = [v1, v2, ..., vn] is given by the formula:
        * magnitude(v) = sqrt(v1^2 + v2^2 + ... + vn^2)
        * Geometrically, the magnitude represents the distance of the vector from the origin in n-dimensional space.
        */
        template<typename T>
        static double magnitude(const shared_ptr<vector<T>>& vector){
            auto result = 0.0;
            for (auto i = 0; i < vector->size(); i++)
                result += vector->at(i) * vector->at(i);
            return sqrt(result);
        }

        /**
        * Overloaded method to calculate the magnitude (or length) of a vector.
        * @param vector Constant reference to the vector.
        * @return The magnitude of the vector.
        * 
        * The magnitude of a vector v = [v1, v2, ..., vn] is given by the formula:
        * magnitude(v) = sqrt(v1^2 + v2^2 + ... + vn^2)
        * Geometrically, the magnitude represents the distance of the vector from the origin in n-dimensional space.
        */
        template<typename T>
        static double magnitude(const vector<T>& vector){
            auto result = 0.0;
            for (auto i = 0; i < vector.size(); i++)
                result += vector.at(i) * vector.at(i);
            return sqrt(result);
        }

        /**
        * Normalizes a vector.
        * @param vector Reference to a shared pointer to the vector to be normalized.
        * 
        * Normalization of a vector refers to the process of dividing each component of the vector by its magnitude. 
        * After normalization, the magnitude of the vector will be 1.
        */
        template<typename T>
        static void normalize(const shared_ptr<vector<T>>& vector){
            auto magnitude = VectorOperations::magnitude(vector);
            for (auto & i : *vector)
                i /= magnitude;
        }

        /**
        * Overloaded method to normalize a vector.
        * @param vector Reference to the vector to be normalized.
        * 
        * Normalization of a vector refers to the process of dividing each component of the vector by its magnitude. 
        * After normalization, the magnitude of the vector will be 1.
        */
        template<typename T>
        static void normalize(vector<T>& vector){
            auto magnitude = VectorOperations::magnitude(vector);
            for (auto & i : vector)
                i /= magnitude;
        }

        /**
        * Calculates the Euclidean distance between two vectors.
        * @param vector1 Constant reference to a shared pointer to the first vector.
        * @param vector2 Constant reference to a shared pointer to the second vector.
        * @return The distance between the two vectors.
        * @throws invalid_argument If the input vectors are of different sizes.
        * 
        * The distance between two vectors v and w is defined as the magnitude of their difference.
        * distance(v, w) = magnitude(v - w)
        */
        template<typename T>
        static double distance(const shared_ptr<vector<T>>& vector1, const shared_ptr<vector<T>>& vector2){
            auto result = 0.0;
            if (vector1->size() != vector2->size())
                throw invalid_argument("Vectors must have the same size");
            for (auto i = 0; i < vector1->size(); i++)
                result += pow(vector1->at(i) - vector2->at(i), 2);
            return sqrt(result);
        }

        /**
        * Overloaded method to calculate the Euclidean distance between two vectors.
        * @param vector1 Constant reference to the first vector.
        * @param vector2 Constant reference to the second vector.
        * @return The distance between the two vectors.
        * @throws invalid_argument If the input vectors are of different sizes.
        * 
        * The distance between two vectors v and w is defined as the magnitude of their difference.
        * distance(v, w) = magnitude(v - w)
        */
        template<typename T>
        static double distance(const vector<T>& vector1, const vector<T>& vector2){
            auto result = 0.0;
            if (vector1.size() != vector2.size())
                throw invalid_argument("Vectors must have the same size");
            for (auto i = 0; i < vector1.size(); i++)
                result += pow(vector1.at(i) - vector2.at(i), 2);
            return sqrt(result);
        }


        /**
        * Calculates the angle between two vectors.
        * @param vector1 Constant reference to a shared pointer to the first vector.
        * @param vector2 Constant reference to a shared pointer to the second vector.
        * @return The angle (in radians) between the two vectors.
        * @throws invalid_argument If the input vectors are of different sizes.
        * 
        * The angle θ between two vectors v and w is given by the formula:
        * cos(θ) = (dot(v, w)) / (magnitude(v) * magnitude(w))
        */
        template<typename T>
        static double angle(const shared_ptr<vector<T>>& vector1, const shared_ptr<vector<T>>& vector2){
            if (vector1->size() != vector2->size())
                throw invalid_argument("Vectors must have the same size");
            auto result = 0.0;
            for (auto i = 0; i < vector1->size(); i++)
                result += vector1->at(i) * vector2->at(i);
            return acos(result / (VectorOperations::magnitude(vector1) * VectorOperations::magnitude(vector2)));
        }

        /**
        * Overloaded method to calculate the angle between two vectors.
        * @param vector1 Constant reference to the first vector.
        * @param vector2 Constant reference to the second vector.
        * @return The angle (in radians) between the two vectors.
        * @throws invalid_argument If the input vectors are of different sizes.
        * 
        * The angle θ between two vectors v and w is given by the formula:
        * cos(θ) = (dot(v, w)) / (magnitude(v) * magnitude(w))
        */
        template<typename T>
        static double angle(const vector<T>& vector1, const vector<T>& vector2){
            if (vector1.size() != vector2.size())
                throw invalid_argument("Vectors must have the same size");
            auto result = 0.0;
            for (auto i = 0; i < vector1.size(); i++)
                result += vector1.at(i) * vector2.at(i);
            return acos(result / (VectorOperations::magnitude(vector1) * VectorOperations::magnitude(vector2)));
        }

        /**
        * Compares two vectors for equality.
        * @param vector1 Constant reference to a shared pointer to the first vector.
        * @param vector2 Constant reference to a shared pointer to the second vector.
        * @return True if the vectors are equal component-wise, otherwise false.
        * @throws invalid_argument If the input vectors are of different sizes.
        * 
        * Two vectors are considered equal if their corresponding components are the same.
        */
        template<typename T>
        static bool areEqualVectors(const shared_ptr<vector<T>>& vector1, const shared_ptr<vector<T>>& vector2){
            if (vector1->size() != vector2->size())
                return false;
            for (auto i = 0; i < vector1->size(); i++)
                if (vector1->at(i) != vector2->at(i))
                    return false;
            return true;
        }

        /**
        * Overloaded method to compare two vectors for equality.
        * @param vector1 Constant reference to the first vector.
        * @param vector2 Constant reference to the second vector.
        * @return True if the vectors are equal component-wise, otherwise false.
        * @throws invalid_argument If the input vectors are of different sizes.
        * 
        * Two vectors are considered equal if their corresponding components are the same.
        */
        template<typename T>
        static bool areEqualVectors(const vector<T>& vector1, const vector<T>& vector2){
            if (vector1->size() != vector2->size())
                return false;
            for (auto i = 0; i < vector1->size(); i++)
                if (vector1->at(i) != vector2->at(i))
                    return false;
            return true;
        }
        
        /**
        * Calculates the sum of all components of a vector.
        * @param vector Constant reference to a shared pointer to the input vector.
        * @return The sum of all components of the vector.
        * 
        * Given a vector v = [v1, v2, ..., vn], the sum is calculated as:
        * sum(v) = v1 + v2 + ... + vn
        */
        template<typename T>
        static T sum(const shared_ptr<vector<T>>& vector){
            auto result = 0.0;
            for(auto& element : *vector)
                result += element;
            return result;
        }

        /**
        * Overloaded method to calculate the sum of all components of a vector.
        * @param vector Constant reference to the input vector.
        * @return The sum of all components of the vector.
        * 
        * Given a vector v = [v1, v2, ..., vn], the sum is calculated as:
        * sum(v) = v1 + v2 + ... + vn
        */
        template<typename T>
        static T sum(const vector<T>& vector){
            auto result = 0.0;
            for(auto& element : vector)
                result += element;
            return result;
        }

        /**
        * Calculates the average of all components of a vector.
        * @param vector Constant reference to a shared pointer to the input vector.
        * @return The average of all components of the vector.
        * @throws invalid_argument If the vector is empty.
        * 
        * Given a vector v = [v1, v2, ..., vn], the average is calculated as:
        * average(v) = (v1 + v2 + ... + vn) / n
        */
        template<typename T>
        static double average(const shared_ptr<vector<T>>& vector){
            return sum(vector) / static_cast<double>(vector->size());
        }

        /**
        * Overloaded method to calculate the average of all components of a vector.
        * @param vector Constant reference to the input vector.
        * @return The average of all components of the vector.
        * @throws invalid_argument If the vector is empty.
        * 
        * Given a vector v = [v1, v2, ..., vn], the average is calculated as:
        * average(v) = (v1 + v2 + ... + vn) / n
        */
        template<typename T>
        static double average(const vector<T>& vector){
            return sum(vector) / static_cast<double>(vector.size());
        }


        /**
        Calculates the variance of a vector of numbers.
        @param vector Constant reference to a shared pointer to the vector.
        @return The variance of the vector.
        @throws invalid_argument If the input vector is empty.
        
        The variance of a vector is a measure of how spread out its values are. It is calculated as the average of the
        squared differences between each element and the mean of the vector.
        Given a vector x with n elements, the variance is calculated as:
        variance(x) = (1/n) * (sum of (x[i] - mean(x))^2 for i=1 to n)
        */
        template<typename T>
        static double variance(const shared_ptr<vector<T>>& vector){
            auto average = VectorOperations::average(vector);
            auto result = 0.0;
            for(auto& element : *vector)
                result += pow(element - average, 2);
            return result / static_cast<double>(vector->size());
        }

        /**
        Calculates the variance of a vector of numbers.
        The variance of a vector is a measure of how spread out its values are. It is calculated as the average of the
        squared differences between each element and the mean of the vector.
        Given a vector x with n elements, the variance is calculated as:
        variance(x) = (1/n) * (sum of (x[i] - mean(x))^2 for i=1 to n)
        @param vector Constant reference to the vector.
        @return The variance of the vector.
        @throws invalid_argument If the input vector is empty.
        */
        template<typename T>
        static double variance(const vector<T>& vector){
            auto average = VectorOperations::average(vector);
            auto result = 0.0;
            for(auto& element : vector)
                result += pow(element - average, 2);
            return result / static_cast<double>(vector.size());
        }


        /**
        Calculates the standard deviation of a vector of numbers.
        The standard deviation of a vector is the square root of its variance. It is a measure of how spread out its values
        are, but it is expressed in the same units as the original data.
        @param vector Constant reference to a shared pointer to the vector.
        @return The standard deviation of the vector.
        @throws invalid_argument If the input vector is empty.
        */
        template<typename T>
        static double standardDeviation(const shared_ptr<vector<T>>& vector){
            return sqrt(variance(vector));
        }


        /**
        Calculates the standard deviation of a vector of doubles.
        The standard deviation of a vector is the square root of its variance. It is a measure of how spread out its values
        are, but it is expressed in the same units as the original data.
        @param vector Constant reference to the vector.
        @return The standard deviation of the vector.
        @throws invalid_argument If the input vector is empty.
        */
        template<typename T>
        static double standardDeviation(const vector<T>& vector){
            return sqrt(variance(vector));
        }


        /**
        Calculates the covariance between two vectors of doubles.
        @param vector1 Constant reference to a shared pointer to the first vector.
        @param vector2 Constant reference to a shared pointer to the second vector.
        @return The covariance between the two vectors.
        @throws invalid_argument If the input vectors are of different sizes.
        
        The covariance between two vectors is a measure of how they vary together. If the covariance is positive, the values of
        one vector tend to be high when the values of the other vector are high, and low when the values of the other vector are
        low. If the covariance is negative, the values of one vector tend to be high when the values of the other vector are low,
        and vice versa. A covariance of zero means that the vectors are uncorrelated, i.e., their values do not vary together.
        Given two vectors X = [x1, x2, ..., xn] and Y = [y1, y2, ..., yn], their covariance is calculated as:
        cov(X, Y) = 1/n * sum((xi - mean(X)) * (yi - mean(Y))), where mean(X) and mean(Y) are the means of X and Y, respectively.
        If the two vectors have the same length, the formula simplifies to:
        cov(X, Y) = 1/n * dot(X - mean(X), Y - mean(Y)), where dot() is the dot product.
        */
        template<typename T>
        static double covariance(const shared_ptr<vector<T>>& vector1, const shared_ptr<vector<T>>& vector2){
            if (vector1->size() != vector2->size())
                throw invalid_argument("Vectors must have the same size");
            auto average1 = VectorOperations::average(vector1);
            auto average2 = VectorOperations::average(vector2);
            auto result = 0.0;
            for (auto i = 0; i < vector1->size(); i++)
                result += (vector1->at(i) - average1) * (vector2->at(i) - average2);
            return result / static_cast<double>(vector1->size());
        }

        /**
        Calculates the covariance between two vectors of doubles.
        @param vector1 Constant reference to the first vector.
        @param vector2 Constant reference to the second vector.
        @return The covariance between the two vectors.
        @throws invalid_argument If the input vectors are of different sizes.
        
        The covariance between two vectors is a measure of how they vary together. If the covariance is positive, the values of
        one vector tend to be high when the values of the other vector are high, and low when the values of the other vector are
        low. If the covariance is negative, the values of one vector tend to be high when the values of the other vector are low,
        and vice versa. A covariance of zero means that the vectors are uncorrelated, i.e., their values do not vary together.
        Given two vectors X = [x1, x2, ..., xn] and Y = [y1, y2, ..., yn], their covariance is calculated as:
        cov(X, Y) = 1/n * sum((xi - mean(X)) * (yi - mean(Y))), where mean(X) and mean(Y) are the means of X and Y, respectively.
        If the two vectors have the same length, the formula simplifies to:
        cov(X, Y) = 1/n * dot(X - mean(X), Y - mean(Y)), where dot() is the dot product.
        */
        template<typename T>
        static double covariance(const vector<T>& vector1, const vector<T>& vector2){
            if (vector1.size() != vector2.size())
                throw invalid_argument("Vectors must have the same size");
            auto average1 = VectorOperations::average(vector1);
            auto average2 = VectorOperations::average(vector2);
            auto result = 0.0;
            for (auto i = 0; i < vector1.size(); i++)
                result += (vector1.at(i) - average1) * (vector2.at(i) - average2);
            return result / static_cast<double>(vector1.size());
        }


        /**
        Calculates the correlation coefficient between two vectors of doubles.
        @param vector1 Constant reference to a shared pointer to the first vector.
        @param vector2 Constant reference to a shared pointer to the second vector.
        @return The correlation coefficient between the two vectors.
        @throws invalid_argument If the input vectors are of different sizes.
        
        The correlation coefficient between two vectors is a measure of how strong the linear relationship is between them. It
        ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no correlation.
        Given two vectors X = [x1, x2, ..., xn] and Y = [y1, y2, ..., yn], their correlation coefficient is calculated as:
        cor(X, Y) = cov(X, Y) / (sd(X) * sd(Y)), where cov(X, Y) is the covariance between X and Y, and sd(X) and sd(Y) are the
        standard deviations of X and Y, respectively.
        */
        template<typename T>
        static double correlation(const shared_ptr<vector<T>>& vector1, const shared_ptr<vector<T>>& vector2){
            if (vector1->size() != vector2->size())
                throw invalid_argument("Vectors must have the same size");
            auto covariance = VectorOperations::covariance(vector1, vector2);
            auto standardDeviation1 = VectorOperations::standardDeviation(vector1);
            auto standardDeviation2 = VectorOperations::standardDeviation(vector2);
            return covariance / (standardDeviation1 * standardDeviation2);
        }


        /**
        Calculates the correlation coefficient between two vectors of doubles.
        @param vector1 Constant reference to the first vector.
        @param vector2 Constant reference to the second vector.
        @return The correlation coefficient between the two vectors.
        @throws invalid_argument If the input vectors are of different sizes.
        
        The correlation coefficient between two vectors is a measure of how strong the linear relationship is between them. It
        ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no correlation.
        Given two vectors X = [x1, x2, ..., xn] and Y = [y1, y2, ..., yn], their correlation coefficient is calculated as:
        cor(X, Y) = cov(X, Y) / (sd(X) * sd(Y)), where cov(X, Y) is the covariance between X and Y, and sd(X) and sd(Y) are the
        standard deviations of X and Y, respectively.
        */
        template<typename T>
        static double correlation(const vector<T>& vector1, const vector<T>& vector2){
            if (vector1.size() != vector2.size())
                throw invalid_argument("Vectors must have the same size");
            auto covariance = VectorOperations::covariance(vector1, vector2);
            auto standardDeviation1 = VectorOperations::standardDeviation(vector1);
            auto standardDeviation2 = VectorOperations::standardDeviation(vector2);
            return covariance / (standardDeviation1 * standardDeviation2);
        }


        /**
        Calculates the correlation between two vectors of integers.
        @param vector Constant reference to a shared pointer to the first vector.
        @return The absolute average difference between the elements of the vector./
        AAD = 1/(n - ) * sum(abs(x_i+1 - x_i))
        Suitable for the average distance between the same coordinate component of different nodes.
        */
        template<typename T>
        static double averageAbsoluteDifference(const shared_ptr<vector<T>>& vector){
            auto average = VectorOperations::average(vector);
            auto result = 0.0;
            for(auto& element : *vector)
                result += abs(element - average);
            return result / static_cast<double>(vector->size());
        }

        /**
        Calculates the correlation between two vectors of integers.
        @param vector Constant reference to the first vector.
        @return The absolute average difference between the elements of the vector./
        AAD = 1/(n - ) * sum(abs(x_i+1 - x_i))
        Suitable for the average distance between the same coordinate component of different nodes.
        */
        template<typename T>
        static double averageAbsoluteDifference(const vector<T>& vector){
            auto average = VectorOperations::average(vector);
            auto result = 0.0;
            for(auto& element : vector)
                result += abs(element - average);
            return result / static_cast<double>(vector.size());
        }
        
    }; // VectorOperations
} // LinearAlgebra

#endif //UNTITLED_VECTOROPERATIONS_H
