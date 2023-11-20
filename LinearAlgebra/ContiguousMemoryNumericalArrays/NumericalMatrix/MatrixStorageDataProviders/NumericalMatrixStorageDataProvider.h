//
// Created by hal9000 on 9/2/23.
//

#ifndef UNTITLED_NUMERICALMATRIXSTORAGEDATAPROVIDER_H
#define UNTITLED_NUMERICALMATRIXSTORAGEDATAPROVIDER_H
#include "../../NumericalVector/NumericalVector.h"
#include <unordered_map>
#include <list>
namespace LinearAlgebra {
    
    
    
    template <typename T>
    class NumericalMatrixStorageDataProvider {
    public:
        explicit NumericalMatrixStorageDataProvider(unsigned numberOfRows, unsigned numberOfColumns, NumericalMatrixFormType formType = General,
                                                    unsigned availableThreads = 1) :
        _numberOfRows(numberOfRows), _numberOfColumns(numberOfColumns), _availableThreads(availableThreads), _formType(formType),
        _elementAssignmentRunning(false), _values(nullptr) {
  /*          elementGetters = unordered_map<NumericalMatrixFormType, T&()>();
            elementSetters = unordered_map<NumericalMatrixFormType, T&()>();*/
        }
        
        virtual double sizeInKB() {}
        
        virtual ~NumericalMatrixStorageDataProvider() = default;
        
        shared_ptr<NumericalVector<T>>& getValues(){
            if (_values->empty())
                throw runtime_error("Values vector is empty.");
            return _values;
        }
        
        virtual vector<shared_ptr<NumericalVector<unsigned>>> getSupplementaryVectors(){ return {}; }
        
        const NumericalMatrixStorageType& getStorageType(){
            return _storageType;
        }
        
        
        virtual T&  getElement(unsigned row, unsigned column) {}

        virtual void setElement(unsigned row, unsigned column, T value) {}

        virtual void eraseElement(unsigned row, unsigned column) {}

        virtual shared_ptr<NumericalVector<T>> getRowSharedPtr(unsigned row) {}
        
        virtual shared_ptr<NumericalVector<T>> getColumnSharedPtr(unsigned column) {}

        virtual void getRow(unsigned row, shared_ptr<NumericalVector<T>> &vector) {}
        
        virtual void getColumn(unsigned column, shared_ptr<NumericalVector<T>> &vector) {}

        virtual void initializeElementAssignment() {}

        virtual void finalizeElementAssignment() {}
        
        virtual void deepCopy(NumericalMatrixStorageDataProvider<T> &inputMatrixData) {}
        
        virtual bool areElementsEqual(NumericalMatrixStorageDataProvider<T> &inputMatrixData) {
            return false;
        }
        
        unsigned getAvailableThreads(){
            return _availableThreads;
        }
        
        void setAvailableThreads(unsigned availableThreads){
            _availableThreads = availableThreads;
            
        }
        
        NumericalMatrixFormType getFormType(){
            return _formType;
        }
        

    protected:
        
        shared_ptr<NumericalVector<T> > _values;
        
        NumericalMatrixStorageType _storageType;
        
        unsigned _numberOfRows;
        
        unsigned _numberOfColumns;
        
        bool _elementAssignmentRunning;
        
        unsigned _availableThreads;
        
        NumericalMatrixFormType _formType;
        
/*        unordered_map<NumericalMatrixFormType, T&()> elementGetters;
        
        unordered_map<NumericalMatrixFormType, T&()> elementSetters;*/
        
        ThreadingOperations<T> _threading;
        
        virtual unsigned _numberOfNonZeroElements() {
            return -1;
        }

        virtual unsigned _numberOfZeroElements() {
            return -1;
        }

        virtual double _density() {
            return -1;
        }
    };

} // LinearAlgebra

#endif //UNTITLED_NUMERICALMATRIXSTORAGEDATAPROVIDER_H
