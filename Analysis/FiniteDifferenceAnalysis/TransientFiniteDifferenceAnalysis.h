//
// Created by hal9000 on 10/6/23.
//

#ifndef UNTITLED_TRANSIENTFINITEDIFFERENCEANALYSIS_H
#define UNTITLED_TRANSIENTFINITEDIFFERENCEANALYSIS_H

#include "SteadyStateFiniteDifferenceAnalysis.h"
#include "../../MathematicalEntities/MathematicalProblem/TransientMathematicalProblem.h"

namespace NumericalAnalysis {

    class TransientFiniteDifferenceAnalysis : public FiniteDifferenceAnalysis{
    public:
        TransientFiniteDifferenceAnalysis(const shared_ptr<TransientMathematicalProblem>& mathematicalProblem,
                                          const shared_ptr<Mesh>& mesh,
                                          const shared_ptr<Solver>& solver,
                                          const shared_ptr<FDSchemeSpecs>&schemeSpecs, CoordinateType coordinateSystem = Natural);
        
        void solve() const override;
        
    private:
        void _assembleM();
        void _assembleC();
        void _assembleK();
        void _assembleEffectiveMatrix();
        void _assembleEffectiveRHS();
        shared_ptr<NumericalMatrix<double>> _M;
        shared_ptr<NumericalMatrix<double>> _C;
        shared_ptr<NumericalMatrix<double>> _K;
        shared_ptr<NumericalMatrix<double>> _K_hat;
        shared_ptr<NumericalVector<double>> _RHS_hat;
    };

} // NumericalAnalysis  


#endif //UNTITLED_TRANSIENTFINITEDIFFERENCEANALYSIS_H
