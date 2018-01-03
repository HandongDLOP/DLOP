#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_    value

#include "Operator//Tensorholder.h"

template<typename DTYPE> class Operator;

enum OptimizeDirection {
    MAXIMIZE,
    MINIMIZE
};

template<typename DTYPE>
class Optimizer {
private:
    Operator<DTYPE> *m_pObjectOperator;

    float m_LearningRate;
    int m_OptimizeDirection;  // 1 or -1

    Tensorholder<DTYPE> **m_apTrainableTensor;
    int m_TrainableTensorDegree;

public:
    Optimizer(Operator<DTYPE> *pObjectOperator, float pLearningRate, OptimizeDirection pOptimizeDirection);

    virtual ~Optimizer();

    int              Alloc(Operator<DTYPE> *pObjectOperator, float pLearningRate, OptimizeDirection pOptimizeDirection);

    int              Delete();

    int              AddTrainableData(Tensorholder<DTYPE> *pTrainableTensor);

    int              UpdateVariable();

    // virtual int UpdateVariable(Tensor<DTYPE> *Trainable, Tensor<DTYPE> *Gradient) = 0;
    virtual int      UpdateVariable(Tensorholder<DTYPE> *pTrainableTensor) = 0;

    void             SetLearningRate(float pLearningRate);

    Operator<DTYPE>* GetObjectOperator() const;

    float            GetLearningRate() const;

    int              GetOptimizeDirection() const;
};

#endif  // OPTIMIZER_H_
