#ifndef GRADIENTDESCENTOPTIMIZER_H_
#define GRADIENTDESCENTOPTIMIZER_H_    value

#include "..//Optimizer.h"

template<typename DTYPE> class GradientDescentOptimizer : public Optimizer<DTYPE>{

public:
    GradientDescentOptimizer(NeuralNetwork<DTYPE> *pNeuralNetwork, float pLearningRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pNeuralNetwork, pLearningRate, pOptimizeDirection) {
        std::cout << "GradientDescentOptimizer::GradientDescentOptimizer(Objective<DTYPE> *, float, OptimizeDirection)" << '\n';
    }

    ~GradientDescentOptimizer() {
        std::cout << "GradientDescentOptimizer::~GradientDescentOptimizer()" << '\n';
    }

    virtual int UpdateVariable(Operator<DTYPE> *pTrainableTensor) {

        Tensor<DTYPE> * trainable_data = pTrainableTensor->GetResult();
        Tensor<DTYPE> * gradient       = pTrainableTensor->GetGradient();

        // learning rate 부분 다시 구현할 필요 있음
        float learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        int capacity = trainable_data->GetData()->GetCapacity();

        for(int i = 0 ; i < capacity; i++){
            (*trainable_data)[i] += learning_rate * (*gradient)[i];
            (*gradient)[i]        = 0;
        }

        return TRUE;
    }
};


#endif  // GRADIENTDESCENTOPTIMIZER_H_
