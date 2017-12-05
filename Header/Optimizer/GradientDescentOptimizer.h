#ifndef GRADIENTDESCENTOPTIMIZER_H_
#define GRADIENTDESCENTOPTIMIZER_H_    value

#include "..//Optimizer.h"

template<typename DTYPE>
class GradientDescentOptimizer : public Optimizer<DTYPE>{
private:
    /* data */

public:
    GradientDescentOptimizer(Operator<DTYPE> *pObjectOperator, float pLearningRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pObjectOperator, pLearningRate, pOptimizeDirection) {
        std::cout << "GradientDescentOptimizer::GradientDescentOptimizer(Operator<DTYPE> *, float, OptimizeDirection)" << '\n';
    }

    ~GradientDescentOptimizer() {
        std::cout << "GradientDescentOptimizer::~GradientDescentOptimizer()" << '\n';
    }

    bool UpdateVariable(TrainableData<DTYPE> *pTrainableData) {
        // std::cout << "GradientDescentOptimizer::UpdateVariable(TrainableData *)" << '\n';

        int *shape = pTrainableData->Data->GetShape();

        DTYPE *****trainable_data = pTrainableData->Data->GetData();
        DTYPE *****gradient       = pTrainableData->Gradient->GetData();

        // learning rate 부분 다시 구현할 필요 있음
        float learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        // average 학습
        // int input_batch = GetBatch();

        int Time    = shape[0];
        int Batch   = shape[1];
        int Channel = shape[2];
        int Row     = shape[3];
        int Col     = shape[4];

        for (int ti = 0; ti < Time; ti++) {
            for (int ba = 0; ba < Batch; ba++) {
                for (int ch = 0; ch < Channel; ch++) {
                    for (int ro = 0; ro < Row; ro++) {
                        for (int co = 0; co < Col; co++) {
                            trainable_data[ti][ba][ch][ro][co] += learning_rate * gradient[ti][ba][ch][ro][co];
                            gradient[ti][ba][ch][ro][co]        = 0;
                        }
                    }
                }
            }
        }
        return true;
    }
};


#endif  // GRADIENTDESCENTOPTIMIZER_H_
