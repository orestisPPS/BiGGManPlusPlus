//
// Created by hal9000 on 9/1/23.
//

#ifndef UNTITLED_THREADINGOPERATIONS_H
#define UNTITLED_THREADINGOPERATIONS_H


#include <stdexcept>
#include <memory>
#include <thread>
#include "../LinearAlgebra/ParallelizationMethods.h"
using namespace LinearAlgebra;
using namespace std;


template<typename T>
class ThreadingOperations {

public:
    
    ThreadingOperations(ParallelizationMethod parallelizationMethod, unsigned availableThreads) :
    _parallelizationMethod(parallelizationMethod), _availableThreads(availableThreads){}
    
    
    const ParallelizationMethod &getParallelizationMethod() {
        return _parallelizationMethod;
    }
    
    void setParallelizationMethod(ParallelizationMethod parallelizationMethod) {
        _parallelizationMethod = parallelizationMethod;
    }
    
    const unsigned& getAvailableThreads() const {
        return _availableThreads;
    }
    
    void setAvailableThreads(unsigned availableThreads) {
        _availableThreads = availableThreads;
    }
    
    template<typename ThreadJob>
    void executeParallelJob(size_t size, ThreadJob task, unsigned cacheLineSize = 64) {
        unsigned doublesPerCacheLine = cacheLineSize / sizeof(T);
        unsigned int numThreads = std::min(_availableThreads, static_cast<unsigned>(size));

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
    static T executeParallelJobWithReduction(size_t size, ThreadJob task, unsigned availableThreads, unsigned cacheLineSize = 64) {
        unsigned doublesPerCacheLine = cacheLineSize / sizeof(T);
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
     * \param _availableThreads The number of threads available for processing.
    * \param cacheLineSize An optional parameter to adjust for system's cache line size (default is 64 bytes).
    * 
    * \return The result vector after the reduction step.
    */
    template<typename ThreadJob>
    vector<T> executeParallelJobWithIncompleteReduction(size_t size, ThreadJob task, unsigned cacheLineSize = 64) {
        unsigned doublesPerCacheLine = cacheLineSize / sizeof(T);
        unsigned int numThreads = std::min(_availableThreads, static_cast<unsigned>(size));

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
    void executeParallelJob(ThreadJob task, unsigned int size) {
        if (_parallelizationMethod == SingleThread) {
            executeParallelJob(size, task, 1);
        } else if (_parallelizationMethod == MultiThread) {
            executeParallelJob(size, task, _availableThreads);
        }
    }

    template<typename ThreadJob>
    double executeParallelJobWithReductionForDoubles(ThreadJob task, unsigned int size) {
        if (_parallelizationMethod == SingleThread) {
            return executeParallelJobWithReduction(size, task, 1);
        } else if (_parallelizationMethod == MultiThread) {
            return executeParallelJobWithReduction(size, task, _availableThreads);
        }
    }
    
    private:
    ParallelizationMethod _parallelizationMethod;
    
    unsigned _availableThreads;
};

#endif //UNTITLED_THREADINGOPERATIONS_H
