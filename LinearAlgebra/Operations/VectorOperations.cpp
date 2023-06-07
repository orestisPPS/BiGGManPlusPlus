//
// Created by hal9000 on 4/14/23.
//

#include "VectorOperations.h"

namespace LinearAlgebra {
    
        double VectorOperations::dotProduct(vector<double> *vector1, vector<double> *vector2) {
            auto result = 0.0;
            if (vector1->size() != vector2->size())
                throw std::invalid_argument("Vectors must have the same size");
            for (auto i = 0; i < vector1->size(); i++)
                result += vector1->at(i) * vector2->at(i);
            return result;
        }
        
        double VectorOperations::dotProduct(vector<double> &vector1, vector<double> &vector2) {
            auto result = 0.0;
            if (vector1.size() != vector2.size())
                throw std::invalid_argument("Vectors must have the same size");
            for (auto i = 0; i < vector1.size(); i++)
                result += vector1.at(i) * vector2.at(i);
            return result;
        }
        
        int VectorOperations::dotProduct(vector<int> *vector1, vector<int> *vector2) {
            auto result = 0;
            if (vector1->size() != vector2->size())
                throw std::invalid_argument("Vectors must have the same size");
            for (auto i = 0; i < vector1->size(); i++)
                result += vector1->at(i) * vector2->at(i);
            return result;
        }
        
        int VectorOperations::dotProduct(vector<int> &vector1, vector<int> &vector2) {
            auto result = 0;
            if (vector1.size() != vector2.size())
                throw std::invalid_argument("Vectors must have the same size");
            for (auto i = 0; i < vector1.size(); i++)
                result += vector1.at(i) * vector2.at(i);
            return result;
        }
        
        vector<double> VectorOperations::crossProduct(vector<double>* vector1, vector<double>* vector2) {
            switch (vector1->size()) {
                case 2:
                    if (vector2->size() != 2)
                        throw std::invalid_argument("Vectors must have 2 dimensions");
                    return {vector1->at(0) * vector2->at(1) - vector1->at(1) * vector2->at(0)};
                case 3:
                    if (vector2->size() != 3)
                        throw std::invalid_argument("Vectors must have 3 dimensions");
                    return {vector1->at(1) * vector2->at(2) - vector1->at(2) * vector2->at(1),
                            vector1->at(2) * vector2->at(0) - vector1->at(0) * vector2->at(2),
                            vector1->at(0) * vector2->at(1) - vector1->at(1) * vector2->at(0)};
                default:
                    throw std::invalid_argument("Vectors must have 2 or 3 dimensions");
            }
        }
        
        
        vector<double> VectorOperations::crossProduct(vector<double> &vector1, vector<double> &vector2) {
            switch (vector1.size()) {
                case 2:
                    if (vector2.size() != 2)
                        throw std::invalid_argument("Vectors must have 2 dimensions");
                    return {vector1.at(0) * vector2.at(1) - vector1.at(1) * vector2.at(0)};
                case 3:
                    if (vector2.size() != 3)
                        throw std::invalid_argument("Vectors must have 3 dimensions");
                    return {vector1.at(1) * vector2.at(2) - vector1.at(2) * vector2.at(1),
                            vector1.at(2) * vector2.at(0) - vector1.at(0) * vector2.at(2),
                            vector1.at(0) * vector2.at(1) - vector1.at(1) * vector2.at(0)};
                default:
                    throw std::invalid_argument("Vectors must have 2 or 3 dimensions");
            }
        }
        
        bool VectorOperations::areEqualVectors(vector<double>* array1, vector<double>* array2) {
            if (array1->size() != array2->size())
                return false;
            for (auto i = 0; i < array1->size(); i++)
                if (array1->at(i) != array2->at(i))
                    return false;
            return true;
        }
        
        bool VectorOperations::areEqualVectors(vector<double> &array1, vector<double> &array2) {
            if (array1.size() != array2.size())
                return false;
            for (auto i = 0; i < array1.size(); i++)
                if (array1.at(i) != array2.at(i))
                    return false;
            return true;
        }
        
        bool VectorOperations::areEqualVectors(vector<int>* array1, vector<int>* array2) {
            if (array1->size() != array2->size())
                return false;
            for (auto i = 0; i < array1->size(); i++)
                if (array1->at(i) != array2->at(i))
                    return false;
            return true;
        }
        
        bool VectorOperations::areEqualVectors(vector<int> &array1, vector<int> &array2) {
            if (array1.size() != array2.size())
                return false;
            for (auto i = 0; i < array1.size(); i++)
                if (array1.at(i) != array2.at(i))
                    return false;
            return true;
        }
        
        double VectorOperations::sum(vector<double>* vector) {
            auto result = 0.0;
            for(auto& element : *vector)
                result += element;
            return result;
        }
        
        double VectorOperations::sum(vector<double> &vector) {
            auto result = 0.0;
            for(auto& element : vector)
                result += element;
            return result;
        }
        
        int VectorOperations::sum(vector<int>* vector) {
            auto result = 0;
            for(auto& element : *vector)
                result += element;
            return result;
        }
        
        int VectorOperations::sum(vector<int> &vector) {
            auto result = 0;
            for(auto& element : vector)
                result += element;
            return result;
        }
        
        double VectorOperations::average(vector<double>* vector) {
            return sum(vector) / static_cast<double>(vector->size());
        }
        
        double VectorOperations::average(vector<double> &vector) {
            return sum(vector) / static_cast<double>(vector.size());
        }

        double VectorOperations::average(vector<int>* vector) {
            return sum(vector) / static_cast<double>(vector->size());
        }
    
        double VectorOperations::average(vector<int> &vector) {
            return sum(vector) / static_cast<double>(vector.size());
        }
        
        double VectorOperations::variance(vector<double>* vector) {
            auto average = VectorOperations::average(vector);
            auto result = 0.0;
            for(auto& element : *vector)
                result += pow(element - average, 2);
            return result / static_cast<double>(vector->size());
        }

        double VectorOperations::variance(vector<double> &vector) {
            auto average = VectorOperations::average(vector);
            auto result = 0.0;
            for(auto& element : vector)
                result += pow(element - average, 2);
            return result / static_cast<double>(vector.size());
        }

        double VectorOperations::variance(vector<int>* vector) {
            auto average = VectorOperations::average(vector);
            auto result = 0.0;
            for(auto& element : *vector)
                result += pow(element - average, 2);
            return result / static_cast<double>(vector->size());
        }
        
        double VectorOperations::variance(vector<int> &vector) {
            auto average = VectorOperations::average(vector);
            auto result = 0.0;
            for(auto& element : vector)
                result += pow(element - average, 2);
            return result / static_cast<double>(vector.size());
        }
        
        double VectorOperations::standardDeviation(vector<double>* vector) {
            return sqrt(variance(vector));
        }
        
        double VectorOperations::standardDeviation(vector<double> &vector) {
            return sqrt(variance(vector));
        }
        
        double VectorOperations::standardDeviation(vector<int>* vector) {
            return sqrt(variance(vector));
        }
        
        double VectorOperations::standardDeviation(vector<int> &vector) {
            return sqrt(variance(vector));
        }
        
        double VectorOperations::covariance(vector<double>* vector1, vector<double>* vector2) {
            if (vector1->size() != vector2->size())
                throw std::invalid_argument("Vectors must have the same size");
            auto average1 = VectorOperations::average(vector1);
            auto average2 = VectorOperations::average(vector2);
            auto result = 0.0;
            for(auto i = 0; i < vector1->size(); i++)
                result += (vector1->at(i) - average1) * (vector2->at(i) - average2);
            return result / static_cast<double>(vector1->size());
        }
        
        double VectorOperations::covariance(vector<double> &vector1, vector<double> &vector2) {
            if (vector1.size() != vector2.size())
                throw std::invalid_argument("Vectors must have the same size");
            auto average1 = VectorOperations::average(vector1);
            auto average2 = VectorOperations::average(vector2);
            auto result = 0.0;
            for(auto i = 0; i < vector1.size(); i++)
                result += (vector1.at(i) - average1) * (vector2.at(i) - average2);
            return result / static_cast<double>(vector1.size());
        }
        
        double VectorOperations::covariance(vector<int>* vector1, vector<int>* vector2) {
            if (vector1->size() != vector2->size())
                throw std::invalid_argument("Vectors must have the same size");
            auto average1 = VectorOperations::average(vector1);
            auto average2 = VectorOperations::average(vector2);
            auto result = 0.0;
            for(auto i = 0; i < vector1->size(); i++)
                result += (vector1->at(i) - average1) * (vector2->at(i) - average2);
            return result / static_cast<double>(vector1->size());
        }
        
        double VectorOperations::covariance(vector<int> &vector1, vector<int> &vector2) {
            if (vector1.size() != vector2.size())
                throw std::invalid_argument("Vectors must have the same size");
            auto average1 = VectorOperations::average(vector1);
            auto average2 = VectorOperations::average(vector2);
            auto result = 0.0;
            for(auto i = 0; i < vector1.size(); i++)
                result += (vector1.at(i) - average1) * (vector2.at(i) - average2);
            return result / static_cast<double>(vector1.size());
        }
        
        double VectorOperations::correlation(vector<double>* vector1, vector<double>* vector2) {
            if (vector1->size() != vector2->size())
                throw std::invalid_argument("Vectors must have the same size");
            auto covariance = VectorOperations::covariance(vector1, vector2);
            auto standardDeviation1 = VectorOperations::standardDeviation(vector1);
            auto standardDeviation2 = VectorOperations::standardDeviation(vector2);
            return covariance / (standardDeviation1 * standardDeviation2);
        }
        
        double VectorOperations::correlation(vector<double> &vector1, vector<double> &vector2) {
            if (vector1.size() != vector2.size())
                throw std::invalid_argument("Vectors must have the same size");
            auto covariance = VectorOperations::covariance(vector1, vector2);
            auto standardDeviation1 = VectorOperations::standardDeviation(vector1);
            auto standardDeviation2 = VectorOperations::standardDeviation(vector2);
            return covariance / (standardDeviation1 * standardDeviation2);
        }
        
        double VectorOperations::correlation(vector<int>* vector1, vector<int>* vector2) {
            if (vector1->size() != vector2->size())
                throw std::invalid_argument("Vectors must have the same size");
            auto covariance = VectorOperations::covariance(vector1, vector2);
            auto standardDeviation1 = VectorOperations::standardDeviation(vector1);
            auto standardDeviation2 = VectorOperations::standardDeviation(vector2);
            return covariance / (standardDeviation1 * standardDeviation2);
        }
        
        double VectorOperations::correlation(vector<int> &vector1, vector<int> &vector2) {
            if (vector1.size() != vector2.size())
                throw std::invalid_argument("Vectors must have the same size");
            auto covariance = VectorOperations::covariance(vector1, vector2);
            auto standardDeviation1 = VectorOperations::standardDeviation(vector1);
            auto standardDeviation2 = VectorOperations::standardDeviation(vector2);
            return covariance / (standardDeviation1 * standardDeviation2);
        }
        
        double VectorOperations::averageAbsoluteDifference(vector<double>& vector1) {
            if (vector1.size() < 2 || vector1.empty())
                throw std::invalid_argument("Vector must have at least 2 elements");
            auto numOfSpaces = static_cast<double>(vector1.size() - 1);
            auto result = 0.0;
            for(auto i = 0; i < numOfSpaces; i++)
                result += abs(vector1[i + 1] - vector1[i]);
            return result / numOfSpaces;
        }
        
        
        
        
        
        

} // LinearAlgebra