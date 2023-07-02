//
// Created by hal9000 on 4/14/23.
//

#ifndef UNTITLED_VECTOROPERATIONS_H
#define UNTITLED_VECTOROPERATIONS_H

#include "../Array/Array.h"

namespace LinearAlgebra {

    class VectorOperations {
    public:

        /**
        
        Calculates the dot product of two vectors.
        @param vector1 The first vector.
        @param vector2 The second vector.
        @return The dot product of the two vectors.
        @throws std::invalid_argument If the input vectors are of different sizes.
        
        The dot product of two vectors is defined as the sum of the products of their corresponding components.
        Given two vectors v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], their dot product is calculated as:
        dot(v, w) = v1w1 + v2w2 + ... + vn*wn
        Geometrically, the dot product of two vectors gives the cosine of the angle between them multiplied by the magnitudes
        of the vectors. If the dot product is zero, it means the vectors are orthogonal (perpendicular) to each other.
        */
        static double dotProduct(const shared_ptr<vector<double>> &vector1, const shared_ptr<vector<double>> &vector2);
        
        /**
        Calculates the dot product of two vectors.
        @param vector1 The first vector.
        @param vector2 The second vector.
        @return The dot product of the two vectors.
        @throws std::invalid_argument If the input vectors are of different sizes.
        
        The dot product of two vectors is defined as the sum of the products of their corresponding components.
        Given two vectors v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], their dot product is calculated as:
        dot(v, w) = v1w1 + v2w2 + ... + vn*wn
        Geometrically, the dot product of two vectors gives the cosine of the angle between them multiplied by the magnitudes
        of the vectors. If the dot product is zero, it means the vectors are orthogonal (perpendicular) to each other.
        */
        static double dotProduct(vector<double> &vector1, vector<double> &vector2);
        
        
        static int dotProduct(const shared_ptr<vector<int>> &vector1, const shared_ptr<vector<int>> &vector2);
        
        static int dotProduct(vector<int> &vector1, vector<int> &vector2);


        /**
        Calculates the cross product of two 3-dimensional vectors.
        @param vector1 The first vector.
        @param vector2 The second vector.
        @return The cross product of the two vectors.
        @throws std::invalid_argument If the input vectors are not 3-dimensional.
        
        The cross product of two 3-dimensional vectors is a vector that is perpendicular to both of them.
        Given two 3-dimensional vectors v = [v1, v2, v3] and w = [w1, w2, w3], their cross product is calculated as:
        cross(v, w) = [v2w3 - v3w2, v3w1 - v1w3, v1w2 - v2w1]
        */
        static vector<double> crossProduct(const shared_ptr<vector<double>>& vector1, const shared_ptr<vector<double>>& vector2);
        
        /**
        Calculates the cross product of two 3-dimensional vectors.
        @param vector1 The first vector.
        @param vector2 The second vector.
        @return The cross product of the two vectors.
        
        @throws std::invalid_argument If the input vectors are not 3-dimensional.
        The cross product of two 3-dimensional vectors is a vector that is perpendicular to both of them.
        Given two 3-dimensional vectors v = [v1, v2, v3] and w = [w1, w2, w3], their cross product is calculated as:
        cross(v, w) = [v2w3 - v3w2, v3w1 - v1w3, v1w2 - v2w1]
        */
        static vector<double> crossProduct(vector<double>& vector1, vector<double> &vector2);
        
        static bool areEqualVectors(const shared_ptr<vector<double>>& array1, const shared_ptr<vector<double>>& array2);
        
        static bool areEqualVectors(vector<double> &array1, vector<double> &array2);
        
        static bool areEqualVectors(const shared_ptr<vector<int>> & array1, const shared_ptr<vector<int>> & array2);
        
        static bool areEqualVectors(vector<int> &array1, vector<int> &array2);
        
        static double sum(const shared_ptr<vector<double>>& vector);
        
        static double sum(vector<double> &vector);
        
        static int sum(const shared_ptr<vector<int>> & vector);
        
        static int sum(vector<int> &vector);
        
        static double average(const shared_ptr<vector<double>>& vector);
        
        static double average(vector<double> &vector);
        
        static double average(const shared_ptr<vector<int>> & vector);
        
        static double average(vector<int> &vector);
        
        /**
        Calculates the variance of a vector of doubles.
        @param vector The input vector.
        @return The variance of the vector.
        @throws std::invalid_argument If the input vector is empty.
        
        The variance of a vector is a measure of how spread out its values are. It is calculated as the average of the
        squared differences between each element and the mean of the vector.
        Given a vector x with n elements, the variance is calculated as:
        variance(x) = (1/n) * (sum of (x[i] - mean(x))^2 for i=1 to n)
        */
        static double variance(const shared_ptr<vector<double>>& vector);
        
        /**
        Calculates the variance of a vector of doubles.
        @param vector The input vector.
        @return The variance of the vector.
        @throws std::invalid_argument If the input vector is empty.
        
        The variance of a vector is a measure of how spread out its values are. It is calculated as the average of the
        squared differences between each element and the mean of the vector.
        Given a vector x with n elements, the variance is calculated as:
        variance(x) = (1/n) * (sum of (x[i] - mean(x))^2 for i=1 to n)
        */
        static double variance(vector<double> &vector);
        
        /**
        Calculates the variance of a vector of integers.
        @param vector The input vector.
        @return The variance of the vector.
        @throws std::invalid_argument If the input vector is empty.
        
        The variance of a vector is a measure of how spread out its values are. It is calculated as the average of the
        squared differences between each element and the mean of the vector.
        Given a vector x with n elements, the variance is calculated as:
        variance(x) = (1/n) * (sum of (x[i] - mean(x))^2 for i=1 to n)
        */
        static double variance(const shared_ptr<vector<int>> & vector);
        
        /**
        Calculates the variance of a vector of integers.
        @param vector The input vector.
        @return The variance of the vector.
        @throws std::invalid_argument If the input vector is empty.
        
        The variance of a vector is a measure of how spread out its values are. It is calculated as the average of the
        squared differences between each element and the mean of the vector.
        Given a vector x with n elements, the variance is calculated as:
        variance(x) = (1/n) * (sum of (x[i] - mean(x))^2 for i=1 to n)
        */
        static double variance(vector<int> &vector);
        
        /**
        Calculates the standard deviation of a vector of doubles.
        @param vector The input vector.
        @return The standard deviation of the vector.
        @throws std::invalid_argument If the input vector is empty.
        
        The standard deviation of a vector is the square root of its variance. It is a measure of how spread out its values
        are, but it is expressed in the same units as the original data.
        */
        static double standardDeviation(const shared_ptr<vector<double>>& vector);
        
        /**
        Calculates the standard deviation of a vector of doubles.
        @param vector The input vector.
        @return The standard deviation of the vector.
        @throws std::invalid_argument If the input vector is empty.
        
        The standard deviation of a vector is the square root of its variance. It is a measure of how spread out its values
        are, but it is expressed in the same units as the original data.
        */
        static double standardDeviation(vector<double> &vector);
        
        /**
        Calculates the standard deviation of a vector of integers.
        @param vector Pointer to vector of integers.
        @return The standard deviation of the vector.
        @throws std::invalid_argument If the input vector is empty.
        
        The standard deviation of a vector is the square root of its variance. It is a measure of how spread out its values
        are, but it is expressed in the same units as the original data.
        */
        static double standardDeviation(const shared_ptr<vector<int>> & vector);
        
        /**
        Calculates the standard deviation of a vector of integers.
        @param vector Reference to vector of integers.
        @return The standard deviation of the vector.
        @throws std::invalid_argument If the input vector is empty.
        
        The standard deviation of a vector is the square root of its variance. It is a measure of how spread out its values
        are, but it is expressed in the same units as the original data.
        */
        static double standardDeviation(vector<int> &vector);
        
        /**
        Calculates the covariance between two vectors of doubles.
        @param vector1 Pointer to the first vector.
        @param vector2 Pointer to the second vector.
        @return The covariance between the two vectors.
        @throws std::invalid_argument If the input vectors are of different sizes.
        
        The covariance between two vectors is a measure of how they vary together. If the covariance is positive, the values of
        one vector tend to be high when the values of the other vector are high, and low when the values of the other vector are
        low. If the covariance is negative, the values of one vector tend to be high when the values of the other vector are low,
        and vice versa. A covariance of zero means that the vectors are uncorrelated, i.e., their values do not vary together.
        Given two vectors X = [x1, x2, ..., xn] and Y = [y1, y2, ..., yn], their covariance is calculated as:
        cov(X, Y) = 1/n * sum((xi - mean(X)) * (yi - mean(Y))), where mean(X) and mean(Y) are the means of X and Y, respectively.
        If the two vectors have the same length, the formula simplifies to:
        cov(X, Y) = 1/n * dot(X - mean(X), Y - mean(Y)), where dot() is the dot product.
        */
        static double covariance(const shared_ptr<vector<double>>& vector1, const shared_ptr<vector<double>>& vector2);
        
        /**
        Calculates the covariance between two vectors of doubles.
        @param vector1 The first vector.
        @param vector2 The second vector.
        @return The covariance between the two vectors.
        @throws std::invalid_argument If the input vectors are of different sizes.
        
        The covariance between two vectors is a measure of how they vary together. If the covariance is positive, the values of
        one vector tend to be high when the values of the other vector are high, and low when the values of the other vector are
        low. If the covariance is negative, the values of one vector tend to be high when the values of the other vector are low,
        and vice versa. A covariance of zero means that the vectors are uncorrelated, i.e., their values do not vary together.
        Given two vectors X = [x1, x2, ..., xn] and Y = [y1, y2, ..., yn], their covariance is calculated as:
        cov(X, Y) = 1/n * sum((xi - mean(X)) * (yi - mean(Y))), where mean(X) and mean(Y) are the means of X and Y, respectively.
        If the two vectors have the same length, the formula simplifies to:
        cov(X, Y) = 1/n * dot(X - mean(X), Y - mean(Y)), where dot() is the dot product.
        */
        static double covariance(vector<double> &vector1, vector<double> &vector2);
        
        /**
        Calculates the covariance between two vectors of integers.
        @param vector1 Pointer to the first vector.
        @param vector2 Pointer to the second vector.
        @return The covariance between the two vectors.
        
        @throws std::invalid_argument If the input vectors are of different sizes.
        The covariance between two vectors is a measure of how they vary together. If the covariance is positive, the values of
        one vector tend to be high when the values of the other vector are high, and low when the values of the other vector are
        low. If the covariance is negative, the values of one vector tend to be high when the values of the other vector are low,
        and vice versa. A covariance of zero means that the vectors are uncorrelated, i.e., their values do not vary together.
        Given two vectors X = [x1, x2, ..., xn] and Y = [y1, y2, ..., yn], their covariance is calculated as:
        cov(X, Y) = 1/n * sum((xi - mean(X)) * (yi - mean(Y))), where mean(X) and mean(Y) are the means of X and Y, respectively.
        If the two vectors have the same length, the formula simplifies to:
        cov(X, Y) = 1/n * dot(X - mean(X), Y - mean(Y)), where dot() is the dot product.
        */
        static double covariance(const shared_ptr<vector<int>> & vector1, const shared_ptr<vector<int>> & vector2);
        
        /**
        Calculates the covariance between two vectors of integers.
        @param vector1 Reference to the first vector.
        @param vector2 Reference to the second vector.
        @return The covariance between the two vectors.
        @throws std::invalid_argument If the input vectors are of different sizes.
        
        The covariance between two vectors is a measure of how they vary together. If the covariance is positive, the values of
        one vector tend to be high when the values of the other vector are high, and low when the values of the other vector are
        low. If the covariance is negative, the values of one vector tend to be high when the values of the other vector are low,
        and vice versa. A covariance of zero means that the vectors are uncorrelated, i.e., their values do not vary together.
        Given two vectors X = [x1, x2, ..., xn] and Y = [y1, y2, ..., yn], their covariance is calculated as:
        cov(X, Y) = 1/n * sum((xi - mean(X)) * (yi - mean(Y))), where mean(X) and mean(Y) are the means of X and Y, respectively.
        If the two vectors have the same length, the formula simplifies to:
        cov(X, Y) = 1/n * dot(X - mean(X), Y - mean(Y)), where dot() is the dot product.
        */
        static double covariance(vector<int> &vector1, vector<int> &vector2);
        
        /**
        Calculates the correlation coefficient between two vectors of doubles.
        @param vector1 Pointer to the first vector.
        @param vector2 Pointer to the second vector.
        @return The correlation coefficient between the two vectors.
        @throws std::invalid_argument If the input vectors are of different sizes.
        
        The correlation coefficient between two vectors is a measure of how strong the linear relationship is between them. It
        ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no correlation.
        Given two vectors X = [x1, x2, ..., xn] and Y = [y1, y2, ..., yn], their correlation coefficient is calculated as:
        cor(X, Y) = cov(X, Y) / (sd(X) * sd(Y)), where cov(X, Y) is the covariance between X and Y, and sd(X) and sd(Y) are the
        standard deviations of X and Y, respectively.
        */
        static double correlation(const shared_ptr<vector<double>> &vector1, const shared_ptr<vector<double>>& vector2);
        
        /**
        Calculates the correlation coefficient between two vectors of doubles.
        @param vector1 Reference to the first vector.
        @param vector2 Reference to the second vector.
        @return The correlation coefficient between the two vectors.
        @throws std::invalid_argument If the input vectors are of different sizes.
        
        The correlation coefficient between two vectors is a measure of how strong the linear relationship is between them. It
        ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no correlation.
        Given two vectors X = [x1, x2, ..., xn] and Y = [y1, y2, ..., yn], their correlation coefficient is calculated as:
        cor(X, Y) = cov(X, Y) / (sd(X) * sd(Y)), where cov(X, Y) is the covariance between X and Y, and sd(X) and sd(Y) are the
        standard deviations of X and Y, respectively.
        */
        static double correlation(vector<double> &vector1, vector<double> &vector2);
        
        /**
        Calculates the correlation coefficient between two vectors of integers.
        @param vector1 Pointer to the first vector.
        @param vector2 Pointer to the second vector.
        @return The correlation coefficient between the two vectors.
        @throws std::invalid_argument If the input vectors are of different sizes.
        
        The correlation coefficient between two vectors is a measure of how strong the linear relationship is between them. It
        ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no correlation.
        Given two vectors X = [x1, x2, ..., xn] and Y = [y1, y2, ..., yn], their correlation coefficient is calculated as:
        cor(X, Y) = cov(X, Y) / (sd(X) * sd(Y)), where cov(X, Y) is the covariance between X and Y, and sd(X) and sd(Y) are the
        standard deviations of X and Y, respectively.
        */
        static double correlation(const shared_ptr<vector<int>> & vector1, const shared_ptr<vector<int>> & vector2);
        
        /**
        Calculates the correlation between two vectors of integers.
        @param vector1 Reference to the first vector.
        @param vector2 Reference to the second vector.
        @return The correlation between the two vectors.
        @throws std::invalid_argument If the input vectors are of different sizes.
        
        The correlation between two vectors is a measure of the strength and direction of their linear relationship. If the correlation
        is positive, the values of one vector tend to increase as the values of the other vector increase, and vice versa. If the correlation
        is negative, the values of one vector tend to decrease as the values of the other vector increase, and vice versa. A correlation of
        zero means that there is no linear relationship between the vectors. The correlation coefficient is a number between -1 and 1, where
        -1 indicates a perfect negative correlation, 0 indicates no correlation, and 1 indicates a perfect positive correlation.
        Given two vectors X = [x1, x2, ..., xn] and Y = [y1, y2, ..., yn], their correlation is calculated as:
        corr(X, Y) = cov(X, Y) / (stdDev(X) * stdDev(Y)), where cov(X, Y) is the covariance between X and Y, and stdDev(X) and stdDev(Y)
        are the standard deviations of X and Y, respectively.
        */
        static double correlation(vector<int> &vector1, vector<int> &vector2);
        
        /**
        Calculates the correlation between two vectors of integers.
        @param vector1 Reference to the vector.
        @return The absolute average difference between the elements of the vector./
         AAD = 1/(n - ) * sum(abs(x_i+1 - x_i))
         Suitable for the average distance between the same coordinate component of different nodes.
        */
        static double averageAbsoluteDifference(vector<double>& vector1);

    
    };

} // LinearAlgebra

#endif //UNTITLED_VECTOROPERATIONS_H
