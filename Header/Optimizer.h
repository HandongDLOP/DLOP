#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_    value

#include "Operator//Placeholder.h"
#include "Operator//Tensorholder.h"

#include "Operator//Reshape.h"

#include "Operator//Relu.h"
#include "Operator//Sigmoid.h"

#include "Operator//Add.h"
#include "Operator//Addconv.h"
#include "Operator//MatMul.h"
#include "Operator//Convolution.h"
#include "Operator//Maxpooling.h"

template<typename DTYPE> class Operator;

enum OptimizeDirection {
    MAXIMIZE,
    MINIMIZE
};

template<typename DTYPE> class Optimizer {
private:
    float m_LearningRate;
    int   m_OptimizeDirection; // 1 or -1

    Operator<DTYPE> **m_apTrainableTensor;
    int m_TrainableTensorDegree;

public:
    Optimizer(float pLearningRate, OptimizeDirection pOptimizeDirection);

    virtual ~Optimizer();

    // ===============

    int Alloc(float pLearningRate, OptimizeDirection pOptimizeDirection);

    int Delete();

    int AddTrainableData(Operator<DTYPE> *pTrainableTensor);

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
