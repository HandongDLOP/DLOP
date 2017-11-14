#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_    value

#include "Tensor.h"

class Optimizer {
private:
    /* data */
    // momentum이나 이런 애들은 따로 변수를 가지고 있어야 한다.

    float m_learning_rate = 0.0;

public:
    Optimizer() {
        std::cout << "Optimizer::Optimizer()" << '\n';
    }

    virtual ~Optimizer() {
        std::cout << "Optimizer::~Optimizer()" << '\n';
    }

    virtual bool UpdateWeight(Tensor *Trainable, Tensor *Gradient) = 0;

    float GetLearningRate(){
        return m_learning_rate;
    }
};

#endif  // OPTIMIZER_H_
