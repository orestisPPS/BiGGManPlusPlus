#include "VectorNorm.h"

namespace LinearAlgebra {

    VectorNorm::VectorNorm(const shared_ptr<vector<double>>& vector, VectorNormType normType, unsigned short lP_Order)
            : _normType(normType) {
        switch (_normType) {
            case L1:
                _value = _calculateL1Norm(vector);
                break;
            case L2:
                _value = _calculateL2Norm(vector);
                break;
            case LInf:
                _value = _calculateLInfNorm(vector);
                break;
            case Lp:
                _value = _calculateLpNorm(vector, lP_Order);
                break;
            default:
                throw std::invalid_argument("Invalid norm type.");
        }
    }

    VectorNormType& VectorNorm::type() {
        return _normType;
    }

    double VectorNorm::value() const {
        return _value;
    }

    double VectorNorm::_calculateL1Norm(const std::shared_ptr<std::vector<double>>& vector) {
        double norm = 0.0;
        for (const auto& component : *vector) {
            norm += std::abs(component);
        }
        return norm;
    }

    double VectorNorm::_calculateL2Norm(const std::shared_ptr<std::vector<double>>& vector) {
        double norm = 0.0;
        for (const auto& component : *vector) {
            norm += std::pow(component, 2);
        }
        return std::sqrt(norm);
    }

    double VectorNorm::_calculateLInfNorm(const std::shared_ptr<std::vector<double>>& vector) {
        double norm = 0.0;
        for (const auto& component : *vector) {
            norm = std::max(norm, std::abs(component));
        }
        return norm;
    }

    double VectorNorm::_calculateLpNorm(const std::shared_ptr<std::vector<double>>& vector, double order) {
        double norm = 0.0;
        for (const auto& component : *vector) {
            norm += std::pow(std::abs(component), order);
        }
        return std::pow(norm, 1.0 / order);
    }

} // LinearAlgebra