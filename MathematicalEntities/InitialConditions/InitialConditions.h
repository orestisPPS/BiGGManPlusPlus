//
// Created by hal9000 on 10/6/23.
//

#ifndef UNTITLED_INITIALCONDITIONS_H
#define UNTITLED_INITIALCONDITIONS_H
#include <unordered_map>
#include <memory>
using namespace std;

namespace MathematicalEntities {

    class InitialConditions {
    public:
        explicit InitialConditions(double initialConditionOrder1);
        
        InitialConditions(double initialConditionOrder1, double initialConditionOrder2);
        
        explicit InitialConditions(unique_ptr<unordered_map<unsigned, double>> initialConditionsOrder1);
        
        InitialConditions(unique_ptr<unordered_map<unsigned, double>> initialConditionsOrder1,
                          unique_ptr<unordered_map<unsigned, double>> initialConditionsOrder2);

        ~InitialConditions();
        
        void setConstantInitialConditions(double initialConditionOrder1);
        
        void setConstantInitialConditions(double initialConditionOrder1, double initialConditionOrder2);
        
        void setVaryingInitialConditions(unique_ptr<unordered_map<unsigned, double>> initialConditionsOrder1);
        
        void setVaryingInitialConditions(unique_ptr<unordered_map<unsigned, double>> initialConditionsOrder1,
                                         unique_ptr<unordered_map<unsigned, double>> initialConditionsOrder2);
        
        void setVaryingInitialConditions(unsigned id, double initialConditionOrder1);
        
        void setVaryingInitialConditions(unsigned id, double initialConditionOrder1, double initialConditionOrder2);
        
        double getInitialCondition(unsigned derivativeOrder) const;
        
        double getInitialCondition(unsigned derivativeOrder, unsigned id);
        
        void deallocate();
        
    private:
        double _constantZeroOrderIC;
        double _constantFirstOrderIC;
        shared_ptr<unordered_map<unsigned, double>> _varyingZeroOrderICs;
        shared_ptr<unordered_map<unsigned, double>> _varyingFirstOrderICs;
        
    };

} // MathematicalEntities

#endif //UNTITLED_INITIALCONDITIONS_H
