#ifndef GRADIENTDESCENTOPTIMIZER_H_
#define GRADIENTDESCENTOPTIMIZER_H_    value

#include "..//Optimizer.h"

template<typename DTYPE>
class GradientDescentOptimizer : public Optimizer<DTYPE>{

public:
    GradientDescentOptimizer(Operator<DTYPE> *pObjectOperator, float pLearningRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pObjectOperator, pLearningRate, pOptimizeDirection) {
        std::cout << "GradientDescentOptimizer::GradientDescentOptimizer(Operator<DTYPE> *, float, OptimizeDirection)" << '\n';
    }

    ~GradientDescentOptimizer() {
        std::cout << "GradientDescentOptimizer::~GradientDescentOptimizer()" << '\n';
    }

    virtual int UpdateVariable(TrainableData<DTYPE> *pTrainableData) {

        return 1;
    }
};


#endif  // GRADIENTDESCENTOPTIMIZER_H_
