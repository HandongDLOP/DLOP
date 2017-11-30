#ifndef GRADIENTDESCENTOPTIMIZER_H_
#define GRADIENTDESCENTOPTIMIZER_H_    value

#include "..//Optimizer.h"

class GradientDescentOptimizer : public Optimizer {
private:
    /* data */

public:
    GradientDescentOptimizer(Operator *pObjectOperator, float pLearningRate, OptimizeDirection pOptimizeDirection) : Optimizer(pObjectOperator, pLearningRate, pOptimizeDirection) {
        std::cout << "GradientDescentOptimizer::GradientDescentOptimizer(Operator *, float, OptimizeDirection)" << '\n';
    }

    virtual ~GradientDescentOptimizer() {
        std::cout << "GradientDescentOptimizer::~GradientDescentOptimizer()" << '\n';
    }

    virtual bool UpdateVariable(TrainableData *pTrainableData) {
        // std::cout << "GradientDescentOptimizer::UpdateVariable(TrainableData *)" << '\n';

        int *shape = pTrainableData->Data->GetShape();

        double *****trainable_data = pTrainableData->Data->GetData();
        double *****gradient       = pTrainableData->Gradient->GetData();

        // learning rate 부분 다시 구현할 필요 있음
        float learning_rate = GetOptimizeDirection() * GetLearningRate();

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
