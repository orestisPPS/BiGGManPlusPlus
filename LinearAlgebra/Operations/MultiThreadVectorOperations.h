//
// Created by hal9000 on 7/28/23.
//

#ifndef UNTITLED_MULTITHREADVECTOROPERATIONS_H
#define UNTITLED_MULTITHREADVECTOROPERATIONS_H

#include <thread>
#include <stdexcept>
#include <vector>
#include <valarray>

using namespace std;

namespace LinearAlgebra {

    /**
     * \class MultiThreadVectorOperations
     * \brief A utility class that provides multi-threaded vector operations.
     *
     * This class offers optimized and parallelized implementations of common vector operations. It internally uses 
     * C++11 threading to distribute the operations across available CPU cores. (VTEC KICKS IN YOOOOOOOOOOOOOOOOOOOOOOOOOO)
    */
    class MultiThreadVectorOperations {
    private:


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
        * \param cacheLineSize An optional parameter to adjust for system's cache line size (default is 64 bytes).
        */
        template<typename ThreadJob>
        static void executeInParallel(size_t size, ThreadJob task, unsigned cacheLineSize = 64) {
            
            // Determine the number of doubles that fit in a cache line.
            // A cache line is the smallest amount of cache that can be loaded and stored from memory.
            // By calculating doublesPerCacheLine, we're determining how many double values we can efficiently
            // load/store at once
            // This is used to ensure that each thread operates on a distinct cache line. 
            unsigned doublesPerCacheLine = cacheLineSize / sizeof(double);
            unsigned int numThreads = thread::hardware_concurrency();
            
            // The size + numThreads - 1 expression is used to round up the division result to ensure that all data is
            // covered even when size is not a multiple of numThreads
            unsigned blockSize = (size + numThreads - 1) / numThreads;
            
            //This is done to ensure that each block of data processed by a thread aligns with the cache lines
            // This is done to avoid false sharing, which is when two threads are accessing the same cache line
            // False sharing can cause performance issues because the cache line has to be reloaded/stored
            blockSize = (blockSize + doublesPerCacheLine - 1) / doublesPerCacheLine * doublesPerCacheLine;

            vector<thread> threads;
            for (unsigned int i = 0; i < numThreads; ++i) {
                unsigned start = i * blockSize;
                unsigned end = start + blockSize;
                if (start >= size) break;
                threads.push_back(thread(task, start, end));
            }

            for (auto& thread : threads) {
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
        * \param cacheLineSize An optional parameter to adjust for system's cache line size (default is 64 bytes).
        * 
        * \return The combined result after the reduction step.
        */
        template<typename T, typename ThreadJob>
        static T executeInParallelWithReduction(size_t size, ThreadJob task, unsigned cacheLineSize = 64) {
            unsigned doublesPerCacheLine = cacheLineSize / sizeof(double);
            unsigned int numThreads = thread::hardware_concurrency();

            unsigned blockSize = (size + numThreads - 1) / numThreads;
            blockSize = (blockSize + doublesPerCacheLine - 1) / doublesPerCacheLine * doublesPerCacheLine;

            vector<T> localResults(numThreads);
            vector<thread> threads;

            for (unsigned int i = 0; i < numThreads; ++i) {
                unsigned start = i * blockSize;
                unsigned end = start + blockSize;
                if (start >= size) break;
                threads.push_back(thread([&](unsigned start, unsigned end, unsigned idx) {
                    localResults[idx] = task(start, end);
                }, start, end, i));
            }

            for (auto& thread : threads) {
                thread.join();
            }

            T finalResult = 0;
            for (T val : localResults) {
                finalResult += val;
            }
            return finalResult;
        }

    public:

        /**
     
        * \brief   * Performs element-wise addition of two vectors using multiple threads.
        * Given two vectors v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], their addition is:
        * add(v, w) = [v1+w1, v2+w2, ..., vn+wn]. The addition is performed in parallel across multiple threads.
        *
        * \tparam T Data type of the vectors.
        * \param a Pointer to the first vector.
        * \param b Pointer to the second vector.
        * \param result Pointer to the result vector.
        * \param size Size of the vectors.
        * \param cacheLineSize Size of the cache line (default is 64 bytes).
        */
        template<typename T>
        static void add(const T* a, const T* b, T* result, size_t size, double cacheLineSize = 64){
            auto additionThreadJob = [&](unsigned start, unsigned end) {
                for (unsigned i = start; i < end && i < size; ++i) {
                    result[i] = a[i] + b[i];
                }
            };
            executeInParallel(size, additionThreadJob, cacheLineSize);
        }

        template<typename T, typename S>
        static void addScaledVector(const T* a, const T* b, T* result, S scale, size_t size, double cacheLineSize = 64){
            auto addScaledVectorThreadJob = [&](unsigned start, unsigned end) {
                for (unsigned i = start; i < end && i < size; ++i) {
                    result[i] = a[i] + scale * b[i];
                }
            };
            executeInParallel(size, addScaledVectorThreadJob, cacheLineSize);
        }

        /**
        * \brief Performs element-wise subtraction of two vectors using multiple threads.
        * Given two vectors v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], their subtraction is:
        * subtract(v, w) = [v1-w1, v2-w2, ..., vn-wn]. The subtraction is performed in parallel across multiple threads.
        *
        * \tparam T The data type of the vectors (e.g., double, float).
        *
        * \param a Pointer to the first input vector.
        * \param b Pointer to the second input vector.
        * \param result Pointer to the result vector.
        * \param size Size of the vectors.
        * \param cacheLineSize Size of the cache line (default is 64 bytes).
        */
        template<typename T>
        static void subtract(const T* a, const T* b, T* result, size_t size, double cacheLineSize = 64){
            auto subtractionThreadJob = [&](unsigned start, unsigned end) {
                for (unsigned i = start; i < end && i < size; ++i) {
                    result[i] = a[i] - b[i];
                }
            };
            executeInParallel(size, subtractionThreadJob, cacheLineSize);
        }

        /**
        * \brief Performs element-wise subtraction of a scaled vector from another vector using multiple threads.
        * Given two vectors v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], and a scalar factor a, their subtraction is:
        * subtract(v, w) = [v1-a*w1, v2-a*w2, ..., vn-*wn]. The subtraction is performed in parallel across multiple threads.
        *
        * \tparam T The data type of the vectors (e.g., double, float).
        *
        * \param a Pointer to the first input vector.
        * \param b Pointer to the second input vector.
        * \param result Pointer to the result vector.
        * \param size Size of the vectors.
        * \param cacheLineSize Size of the cache line (default is 64 bytes).
        */
        template<typename T, typename S>
        static void subtractScaledVector(const T* a, const T* b, T* result, S scale, size_t size, double cacheLineSize = 64){
            auto subtractScaledVectorThreadJob = [&](unsigned start, unsigned end) {
                for (unsigned i = start; i < end && i < size; ++i) {
                    result[i] = a[i] - scale * b[i];
                }
            };
            executeInParallel(size, subtractScaledVectorThreadJob, cacheLineSize);
        }

        /**
        * \brief Calculates the dot product of 2 vectors using multiple threads.
        * The dot product of two vectors is defined as the sum of the products of their corresponding components.
        * Given two vectors v = [v1, v2, ..., vn] and w = [w1, w2, ..., wn], their dot product is calculated as:
        * dot(v, w) = v1w1 + v2w2 + ... + vn*wn
        * Geometrically, the dot product of two vectors gives the cosine of the angle between them multiplied by the magnitudes
        * of the vectors. If the dot product is zero, it means the vectors are orthogonal (perpendicular) to each other.
        * The subtraction is performed in parallel across multiple threads.
        *
        * \tparam T The data type of the vectors (e.g., double, float).
        * 
        * \param a Pointer to the first input vector.
        * \param b Pointer to the second input vector.
        * \param result Pointer to the output vector where the result will be stored.
        * \param size The number of elements in the vectors.
        * \param cacheLineSize An optional parameter to adjust for system's cache line size (default is 64 bytes).
        */
        template<typename T>
        static T dotProduct(const T* a, const T* b, size_t size, double cacheLineSize = 64) {
            auto dotProductThreadJob = [&](unsigned start, unsigned end) -> T {
                T localDot = 0.0;
                for (unsigned i = start; i < end && i < size; ++i) {
                    localDot += a[i] * b[i];
                }
                return localDot;
            };
            return executeInParallelWithReduction<T>(size, dotProductThreadJob, cacheLineSize);
        }

        /**
        * \brief Deep copy the source vector into the destination vector using multiple threads.
        *
        * This function ensures that each thread copies a distinct segment of the source vector to the destination vector. 
        * The copying operation is performed in parallel across multiple threads.
        *
        * \tparam T The data type of the vectors (e.g., double, float).
        * 
        * \param source Pointer to the source vector.
        * \param destination Pointer to the destination vector.
        * \param size The number of elements in the vectors.
        * \param cacheLineSize An optional parameter to adjust for system's cache line size (default is 64 bytes).
        */
        template<typename T>
        static void deepCopy(const T* source, T* destination, size_t size, unsigned cacheLineSize = 64) {
            auto deepCopyThreadJob = [&](unsigned start, unsigned end) {
                for (unsigned i = start; i < end && i < size; ++i) {
                    destination[i] = source[i];
                }
            };
            executeInParallel(size, deepCopyThreadJob, cacheLineSize);
        }


        /**
        * \brief Multi-threaded matrix-vector multiplication.
        *
        * This function computes the product of a matrix and a vector using multiple threads. Given a matrix A of 
        * dimensions m x n and a vector x of dimensions n, the result of the multiplication Ax is a vector b of 
        * dimensions m.
        *
        * \tparam T The data type of the matrix and vectors (e.g., double, float).
        * 
        * \param A Pointer to the input matrix with dimensions m x n.
        * \param x Pointer to the input vector with dimensions n.
        * \param b Pointer to the output vector where the result will be stored. It should have dimensions m.
        * \param m The number of rows in the matrix.
        * \param n The number of columns in the matrix (also the size of vector x).
        * \param cacheLineSize An optional parameter to adjust for system's cache line size (default is 64 bytes).
        */
        template<typename T>
        static void matrixVectorMultiplication(const T* A, const T* x, T* b, size_t m, size_t n, unsigned cacheLineSize = 64) {
            auto matrixVectorProductThreadJob = [&](unsigned start, unsigned end) {
                for (unsigned i = start; i < end && i < m; ++i) {
                    b[i] = 0.0;
                    for (unsigned j = 0; j < n; ++j) {
                        b[i] += A[i * n + j] * x[j];
                    }
                }
            };
            executeInParallel(m, matrixVectorProductThreadJob, cacheLineSize);
        }
    };
} // LinearAlgebra

#endif //UNTITLED_MULTITHREADVECTOROPERATIONS_H
