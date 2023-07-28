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

    class MultiThreadVectorOperations {
    public:

        template<typename T>
        static double* add(const T* a, const T* b, size_t size, double cacheLineSize = 64){
            
            unsigned doublesPerCacheLine = cacheLineSize / sizeof(T);
            unsigned int numThreads = thread::hardware_concurrency();

            // Calculate the chunk size
            unsigned blockSize = (size + numThreads - 1) / numThreads;
            blockSize = (blockSize + doublesPerCacheLine - 1) / doublesPerCacheLine * doublesPerCacheLine; // Ensure chunk size is a multiple of cache line size
            
            auto result = new double[size];
            
            // Helper lambda function for each thread's job
            auto additionThreadJob = [&](unsigned start, unsigned end) {
                for (unsigned i = start; i < end && i < size; ++i) {
                    result[i] = a[i] + b[i];
                }
            };

            vector<thread> threads;
            for (unsigned int i = 0; i < numThreads; ++i) {
                unsigned start = i * blockSize;
                unsigned end = start + blockSize;
                if (start >= size) break;
                threads.push_back(thread(additionThreadJob, start, end));
            }

            // Join the threads
            for (auto& thread : threads) {
                thread.join();
            }
            return result;
        }

        template<typename T>
        static double* subtract(const T* a, const T* b, size_t size, double cacheLineSize = 64){

            unsigned doublesPerCacheLine = cacheLineSize / sizeof(T);
            unsigned int numThreads = thread::hardware_concurrency();

            // Calculate the block size based on the number of threads
            unsigned blockSize = (size + numThreads - 1) / numThreads;
            blockSize = (blockSize + doublesPerCacheLine - 1) / doublesPerCacheLine * doublesPerCacheLine; // Ensure chunk size is a multiple of cache line size
            
            auto result = new double[size];
            
            // Helper lambda function for each thread's job
            auto subtractionThreadJob = [&](unsigned start, unsigned end) {
                for (unsigned i = start; i < end && i < size; ++i) {
                    result[i] += a[i] - b[i];
                }
            };

            vector<thread> threads;
            for (unsigned int i = 0; i < numThreads; ++i) {
                unsigned start = i * blockSize;
                unsigned end = start + blockSize;
                if (start >= size) break;
                threads.push_back(thread(subtractionThreadJob, start, end));
            }

            // Join the threads
            for (auto& thread : threads) {
                thread.join();
            }
            return result;
        }

        template<typename T>
        static T dotProduct(const T* a, const T* b, size_t size, double cacheLineSize = 64) {
            unsigned doublesPerCacheLine = cacheLineSize / sizeof(T);
            unsigned int numThreads = thread::hardware_concurrency();

            // Calculate the block size
            unsigned blockSize = (size + numThreads - 1) / numThreads;
            blockSize = (blockSize + doublesPerCacheLine - 1) / doublesPerCacheLine * doublesPerCacheLine;

            vector<double> localResults(numThreads, 0.0);

            // Helper lambda function for each thread's job
            auto dotProductThreadJob = [&](unsigned start, unsigned end) {
                double localDot = 0.0;
                for (unsigned i = start; i < end && i < size; ++i) {
                    localDot += a[i] * b[i];
                }
                localResults[start / blockSize] = localDot;
            };

            vector<thread> threads;
            for (unsigned int i = 0; i < numThreads; ++i) {
                unsigned start = i * blockSize;
                unsigned end = start + blockSize;
                if (start >= size) break;
                threads.push_back(thread(dotProductThreadJob, start, end));
            }

            // Join the threads
            for (auto& thread : threads) {
                thread.join();
            }

            // Reduce results
            T finalResult = 0.0;
            for (T val : localResults) {
                finalResult += val;
            }
            return finalResult;
        }

        template<typename T>
        static T sum(const T* a, size_t size, double cacheLineSize = 64) {
            unsigned doublesPerCacheLine = cacheLineSize / sizeof(T);
            unsigned int numThreads = thread::hardware_concurrency();

            // Calculate the block size
            unsigned blockSize = (size + numThreads - 1) / numThreads;
            blockSize = (blockSize + doublesPerCacheLine - 1) / doublesPerCacheLine * doublesPerCacheLine;

            vector<T> localResults(numThreads, 0.0);

            // Helper lambda function for each thread's job
            auto sumThreadJob = [&](unsigned start, unsigned end) {
                T localSum = 0.0;
                for (unsigned i = start; i < end && i < size; ++i) {
                    localSum += a[i];
                }
                localResults[start / blockSize] = localSum;
            };

            vector<thread> threads;
            for (unsigned int i = 0; i < numThreads; ++i) {
                unsigned start = i * blockSize;
                unsigned end = start + blockSize;
                if (start >= size) break;
                threads.push_back(thread(sumThreadJob, start, end));
            }

            // Join the threads
            for (auto& thread : threads) {
                thread.join();
            }

            // Reduce results
            T finalResult = 0.0;
            for (T val : localResults) {
                finalResult += val;
            }
            return finalResult;
        }

    };
} // LinearAlgebra

#endif //UNTITLED_MULTITHREADVECTOROPERATIONS_H
