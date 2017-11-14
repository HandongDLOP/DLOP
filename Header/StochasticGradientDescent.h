#ifndef STOCHASTICGRADIENTDESCENT_H_
#define STOCHASTICGRADIENTDESCENT_H_    value

#include "Optimizer.h"

class StochasticGradientDescent : public Optimizer {
private:
    /* data */

public:
    StochasticGradientDescent() {
        std::cout << "StochasticGradientDescent::StochasticGradientDescent()" << '\n';
    }

    virtual ~StochasticGradientDescent() {
        std::cout << "StochasticGradientDescent::~StochasticGradientDescent()" << '\n';
    }

    virtual bool UpdateWeight(Tensor *pTrainable, Tensor *pGradient) {
        std::cout << "StochasticGradientDescent::UpdateWeight(Tensor *, Tensor *)" << '\n';

        float *trainable_data = pTrainable->GetData();
        float *gradient       = pGradient->GetData();

        int flatdim = pTrainable->GetFlatDim();

        // learning rate 부분 다시 구현할 필요 있음
        float learning_rate = GetLearningRate();
        learning_rate = 0.6;

        for (int i = 0; i < flatdim; i++) {
            trainable_data[i] -= learning_rate * gradient[i];
            gradient[i]        = 0; // gradient 초기화
        }

        return true;
    }
};


#endif  // STOCHASTICGRADIENTDESCENT_H_
