#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_    value

#include "Objective.h"

template<typename DTYPE> class Operator;

enum OptimizeDirection {
    MAXIMIZE,
    MINIMIZE
};

template<typename DTYPE> class Optimizer {
private:
    Objective<DTYPE> *m_pObjectOperator;

    float m_LearningRate;
    int   m_OptimizeDirection; // 1 or -1

    Operator<DTYPE> **m_apTrainableTensor;
    int m_TrainableTensorDegree;

public:
    Optimizer(Objective<DTYPE> *pObjectOperator, float pLearningRate, OptimizeDirection pOptimizeDirection);
    Optimizer(float pLearningRate, OptimizeDirection pOptimizeDirection);

    virtual ~Optimizer();

    // ===============

    int Alloc(Objective<DTYPE> *pObjectOperator, float pLearningRate, OptimizeDirection pOptimizeDirection);
    int Alloc(float pLearningRate, OptimizeDirection pOptimizeDirection);

    int Delete();

    int AddTrainableData(Operator<DTYPE> *pTrainableTensor);

    int AddTrainableTensor(Operator<DTYPE> *pTrainableTensor);


    // ===============
    int         UpdateVariable();

    // virtual int UpdateVariable(Tensor<DTYPE> *Trainable, Tensor<DTYPE> *Gradient) = 0;
    virtual int UpdateVariable(Operator<DTYPE> *pTrainableTensor) = 0;

    // ===============

    void              SetLearningRate(float pLearningRate);

    Objective<DTYPE>* GetObjectOperator() const;

    float             GetLearningRate() const;

    int               GetOptimizeDirection() const;
};

#endif  // OPTIMIZER_H_
