//
// Created by hal9000 on 4/14/23.
//

#include "VectorOperations.h"

namespace LinearAlgebra {


    template<typename T>
    T VectorOperations::dotProduct(const std::shared_ptr<std::vector<T>> &vector1,
                                   const std::shared_ptr<vector<T>> &vector2) {
        auto result = 0.0;
        if (vector1->size() != vector2->size())
            throw std::invalid_argument("Vectors must have the same size");
        for (auto i = 0; i < vector1->size(); i++)
            result += vector1->at(i) * vector2->at(i);
        return result;
    }
    
    template<typename T>
    T VectorOperations::dotProduct(const vector<T>& vector1, const std::shared_ptr<Array<T>>& vector2) {
        auto result = 0.0;
        if (vector1.size() != vector2->size())
            throw std::invalid_argument("Vectors must have the same size");
        for (auto i = 0; i < vector1.size(); i++)
            result += vector1.at(i) * vector2->at(i);
        return result;
    }
    
    template<typename T>
    T VectorOperations::dotProductWithTranspose(const std::shared_ptr<vector<T>>& vector1) {
        if (vector1->empty())
            throw std::invalid_argument("Vector must not be empty");
        auto result = 0.0;
        for (auto i = 0; i < vector1->size(); i++)
            result += vector1->at(i) * vector1->at(i);    
        return result;
    }
    
    template<typename T>
    T VectorOperations::dotProductWithTranspose(const vector<T>& vector1) {
        if (vector1.empty())
            throw std::invalid_argument("Vector must not be empty");
        auto result = 0.0;
        for (auto i = 0; i < vector1.size(); i++)
            result += vector1.at(i) * vector1.at(i);
        return result;
    }
        
    template<typename T>
    vector<T> VectorOperations::crossProduct(const std::shared_ptr<vector<T>>& vector1, const std::shared_ptr<vector<T>>& vector2) {
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
    
    template<typename T>
    vector<T> VectorOperations::crossProduct(const vector<T>& vector1, const vector<T>& vector2) {
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

    template<typename T, typename S>
    void VectorOperations::scale(const std::shared_ptr<vector<T>>& vector, S scalar) {
        for (auto & i : *vector)
            i *= scalar;
    }
    
    template<typename T, typename S>
    void VectorOperations::scale(vector<T>& vector, S scalar) {
        for (auto & i : vector)
            i *= scalar;
    }
    
    template<typename T>
    vector<T> VectorOperations::add(const std::shared_ptr<vector<T>>& vector1, const std::shared_ptr<vector<T>>& vector2) {
        if (vector1->size() != vector2->size())
            throw std::invalid_argument("Vectors must have the same size");
        for (auto i = 0; i < vector1->size(); i++)
            vector1->at(i) += vector2->at(i);
    }
    
    template<typename T>
    vector<T> VectorOperations::add(const vector<T>& vector1, const vector<T>& vector2) {
        if (vector1.size() != vector2.size())
            throw std::invalid_argument("Vectors must have the same size");
        for (auto i = 0; i < vector1.size(); i++)
            vector1.at(i) += vector2.at(i);
    }
    
    template<typename T>
    vector<T> VectorOperations::subtract(const std::shared_ptr<vector<T>>& vector1, const std::shared_ptr<vector<T>>& vector2) {
        if (vector1->size() != vector2->size())
            throw std::invalid_argument("Vectors must have the same size");
        for (auto i = 0; i < vector1->size(); i++)
            vector1->at(i) -= vector2->at(i);
    }
    
    template<typename T>
    vector<T> VectorOperations::subtract(const vector<T>& vector1, const vector<T>& vector2) {
        if (vector1.size() != vector2.size())
            throw std::invalid_argument("Vectors must have the same size");
        for (auto i = 0; i < vector1.size(); i++)
            vector1.at(i) -= vector2.at(i);
    }
    
    template<typename T>
    double VectorOperations::magnitude(const std::shared_ptr<vector<T>>& vector) {
        auto result = 0.0;
        for (auto i = 0; i < vector->size(); i++)
            result += vector->at(i) * vector->at(i);
        return sqrt(result);
    }
    
    template<typename T>
    double VectorOperations::magnitude(const vector<T>& vector) {
        auto result = 0.0;
        for (auto i = 0; i < vector.size(); i++)
            result += vector.at(i) * vector.at(i);
        return sqrt(result);
    }
    
    template<typename T>
    void VectorOperations::normalize(const std::shared_ptr<vector<T>>& vector) {
        auto magnitude = VectorOperations::magnitude(vector);
        for (auto & i : *vector)
            i /= magnitude;
    }
    
    template<typename T>
    void VectorOperations::normalize(vector<T>& vector) {
        auto magnitude = VectorOperations::magnitude(vector);
        for (auto & i : vector)
            i /= magnitude;
    }
    
    
    template<typename T>
    double VectorOperations::distance(const std::shared_ptr<vector<T>>& vector1, const std::shared_ptr<vector<T>>& vector2) {
        auto result = 0.0;
        if (vector1->size() != vector2->size())
            throw std::invalid_argument("Vectors must have the same size");
        for (auto i = 0; i < vector1->size(); i++)
            result += pow(vector1->at(i) - vector2->at(i), 2);
        return sqrt(result);
    }
    
    template<typename T>
    double VectorOperations::distance(const vector<T>& vector1, const vector<T>& vector2) {
        auto result = 0.0;
        if (vector1.size() != vector2.size())
            throw std::invalid_argument("Vectors must have the same size");
        for (auto i = 0; i < vector1.size(); i++)
            result += pow(vector1.at(i) - vector2.at(i), 2);
        return sqrt(result);
    }
    
    template<typename T>
    double VectorOperations::angle(const std::shared_ptr<vector<T>>& vector1, const std::shared_ptr<vector<T>>& vector2) {
        if (vector1->size() != vector2->size())
            throw std::invalid_argument("Vectors must have the same size");
        auto result = 0.0;
        for (auto i = 0; i < vector1->size(); i++)
            result += vector1->at(i) * vector2->at(i);
        return acos(result / (VectorOperations::magnitude(vector1) * VectorOperations::magnitude(vector2)));
    }
    
    template<typename T>
    double VectorOperations::angle(const vector<T>& vector1, const vector<T>& vector2) {
        if (vector1.size() != vector2.size())
            throw std::invalid_argument("Vectors must have the same size");
        auto result = 0.0;
        for (auto i = 0; i < vector1.size(); i++)
            result += vector1.at(i) * vector2.at(i);
        return acos(result / (VectorOperations::magnitude(vector1) * VectorOperations::magnitude(vector2)));
    }
    
    template<typename T>
    bool VectorOperations::areEqualVectors(const std::shared_ptr<vector<T>>& array1, const std::shared_ptr<vector<T>>& array2) {
        if (array1->size() != array2->size())
            return false;
        for (auto i = 0; i < array1->size(); i++)
            if (array1->at(i) != array2->at(i))
                return false;
        return true;
    }
    
    template<typename T>
    bool VectorOperations::areEqualVectors(const vector<T>& array1, const vector<T>& array2) {
        if (array1.size() != array2.size())
            return false;
        for (auto i = 0; i < array1.size(); i++)
            if (array1.at(i) != array2.at(i))
                return false;
        return true;
    }
        
    template<typename T>
    T VectorOperations::sum(const std::shared_ptr<vector<T>>& vector) {
        auto result = 0.0;
        for(auto& element : *vector)
            result += element;
        return result;
    }
    
    template<typename T>
    T VectorOperations::sum(const vector<T> &vector) {
        auto result = 0.0;
        for(auto& element : vector)
            result += element;
        return result;
    }
    

    template<typename T>
    double VectorOperations::average(const std::shared_ptr<vector<T>>& vector) {
        return sum(vector) / static_cast<double>(vector->size());
    }
    
    template<typename T>
    double VectorOperations::average(const vector<T> &vector) {
        return sum(vector) / static_cast<double>(vector.size());
    }
    
        
    template<typename T>
    double VectorOperations::variance(const std::shared_ptr<vector<T>>& vector) {
        auto average = VectorOperations::average(vector);
        auto result = 0.0;
        for(auto& element : *vector)
            result += pow(element - average, 2);
        return result / static_cast<double>(vector->size());
    }
    
    template<typename T>
    double VectorOperations::variance(const vector<T> &vector) {
        auto average = VectorOperations::average(vector);
        auto result = 0.0;
        for(auto& element : vector)
            result += pow(element - average, 2);
        return result / static_cast<double>(vector.size());
    }
    
    template<typename T>
    double VectorOperations::standardDeviation(const std::shared_ptr<vector<T>>& vector) {
        return sqrt(variance(vector));
    }
    
    template<typename T>
    double VectorOperations::standardDeviation(const vector<T> &vector) {
        return sqrt(variance(vector));
    }
    
    template<typename T>
    double VectorOperations::covariance(const std::shared_ptr<vector<T>>& vector1, const std::shared_ptr<vector<T>>& vector2) {
        if (vector1->size() != vector2->size())
            throw std::invalid_argument("Vectors must have the same size");
        auto average1 = VectorOperations::average(vector1);
        auto average2 = VectorOperations::average(vector2);
        auto result = 0.0;
        for (auto i = 0; i < vector1->size(); i++)
            result += (vector1->at(i) - average1) * (vector2->at(i) - average2);
        return result / static_cast<double>(vector1->size());
    }
    
    template<typename T>
    double VectorOperations::covariance(const vector<T> &vector1, const vector<T> &vector2) {
        if (vector1.size() != vector2.size())
            throw std::invalid_argument("Vectors must have the same size");
        auto average1 = VectorOperations::average(vector1);
        auto average2 = VectorOperations::average(vector2);
        auto result = 0.0;
        for (auto i = 0; i < vector1.size(); i++)
            result += (vector1.at(i) - average1) * (vector2.at(i) - average2);
        return result / static_cast<double>(vector1.size());
    }
    
    template<typename T>
    
    double VectorOperations::correlation(const std::shared_ptr<vector<T>>& vector1, const std::shared_ptr<vector<T>>& vector2) {
        if (vector1->size() != vector2->size())
            throw std::invalid_argument("Vectors must have the same size");
        auto covariance = VectorOperations::covariance(vector1, vector2);
        auto standardDeviation1 = VectorOperations::standardDeviation(vector1);
        auto standardDeviation2 = VectorOperations::standardDeviation(vector2);
        return covariance / (standardDeviation1 * standardDeviation2);
    }
    
    template<typename T>
    double VectorOperations::correlation(const vector<T> &vector1, const vector<T> &vector2) {
        if (vector1.size() != vector2.size())
            throw std::invalid_argument("Vectors must have the same size");
        auto covariance = VectorOperations::covariance(vector1, vector2);
        auto standardDeviation1 = VectorOperations::standardDeviation(vector1);
        auto standardDeviation2 = VectorOperations::standardDeviation(vector2);
        return covariance / (standardDeviation1 * standardDeviation2);
    }
    
    template<typename T>
    double VectorOperations::averageAbsoluteDifference(const std::shared_ptr<vector<T>> &vector) {
        auto average = VectorOperations::average(vector);
        auto result = 0.0;
        for(auto& element : *vector)
            result += abs(element - average);
        return result / static_cast<double>(vector->size());
    }
    
    template<typename T>
    double VectorOperations::averageAbsoluteDifference(const vector<T> &vector) {
        auto average = VectorOperations::average(vector);
        auto result = 0.0;
        for(auto& element : vector)
            result += abs(element - average);
        return result / static_cast<double>(vector.size());
    }
        
        
        
        
        

} // LinearAlgebra