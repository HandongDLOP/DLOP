#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_    value

#include "Objective//SoftmaxCrossEntropy.h"
#include "Objective//MSE.h"

template<typename DTYPE> class Operator;

enum OptimizeDirection {
    MAXIMIZE,
    MINIMIZE
};

template<typename DTYPE> class Optimizer {
private:
    Objective<DTYPE> * m_pObjective;

    float m_LearningRate;
    int   m_OptimizeDirection; // 1 or -1

    Operator<DTYPE> **m_apTrainableTensor;
    int m_TrainableTensorDegree;

public:
    Optimizer(Objective<DTYPE> *pObjective, float pLearningRate, OptimizeDirection pOptimizeDirection);

    virtual ~Optimizer();

    // ===============

    int Alloc(Objective<DTYPE> *pObjective, float pLearningRate, OptimizeDirection pOptimizeDirection);

    int Delete();

    int AddTrainableTensor(Objective<DTYPE> *pObjective);
    int AddTrainableTensor(Operator<DTYPE> *pTrainableTensor);

    // ===============
    int         UpdateVariable();

    // virtual int UpdateVariable(Tensor<DTYPE> *Trainable, Tensor<DTYPE> *Gradient) = 0;
    virtual int UpdateVariable(Operator<DTYPE> *pTrainableTensor) = 0;

    // ===============

    void  SetLearningRate(float pLearningRate);

    float GetLearningRate() const;

    int   GetOptimizeDirection() const;
};

#endif  // OPTIMIZER_H_
