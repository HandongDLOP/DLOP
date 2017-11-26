#ifndef STOCHASTICGRADIENTDESCENT_H_
#define STOCHASTICGRADIENTDESCENT_H_    value

#include "Optimizer.h"

class StochasticGradientDescent : public Optimizer {
private:
    /* data */

public:
    StochasticGradientDescent(float pLearningRate) {
        std::cout << "StochasticGradientDescent::StochasticGradientDescent()" << '\n';
        SetLearningRate(pLearningRate);
    }

    virtual ~StochasticGradientDescent() {
        std::cout << "StochasticGradientDescent::~StochasticGradientDescent()" << '\n';
    }

    virtual bool UpdateWeight(Tensor *pTrainable, Tensor *pGradient) {
        std::cout << "StochasticGradientDescent::UpdateWeight(Tensor *, Tensor *)" << '\n';

        int *shape = pTrainable->GetShape();

        double *****trainable_data = pTrainable->GetData();
        double *****gradient       = pGradient->GetData();

        // learning rate 부분 다시 구현할 필요 있음
        float learning_rate = GetLearningRate();

        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            trainable_data[ti][ba][ch][ro][co] -= learning_rate * gradient[ti][ba][ch][ro][co];
                            gradient[ti][ba][ch][ro][co]        = 0;
                        }
                    }
                }
            }
        }

        return true;
    }
};


#endif  // STOCHASTICGRADIENTDESCENT_H_
