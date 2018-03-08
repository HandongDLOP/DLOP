#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Optimizer//GradientDescentOptimizer.h"
#include "Temporary_method.h"

template<typename DTYPE> class NeuralNetwork {
private:
#if __CUDNN__
    cudnnHandle_t m_cudnnHandle;
#endif  // if __CUDNN__
    Placeholder<DTYPE> **m_aaPlaceholder;
    Operator<DTYPE> **m_aaOperator;
    Tensorholder<DTYPE> **m_aaTensorholder;

    int m_PlaceholderDegree;
    int m_OperatorDegree;
    int m_TensorholderDegree;

    Objective<DTYPE> *m_aObjective;
    Optimizer<DTYPE> *m_aOptimizer;

    // Optimizer<DTYPE> *m_aOptimizer;

public:
    NeuralNetwork();
    virtual ~NeuralNetwork();

    // int  Alloc();
    void Delete();

#if 0
    int  CuDNN_DevTensorAlloc(Operator<DTYPE> *pHostTensor);
#endif  // if 0
    // =======

    // 추후 직접 변수를 만들지 않은 operator* + operator*의 변환 변수도 자동으로 할당될 수 있도록 Operator와 NN class를 수정해야 한다.
    Placeholder<DTYPE> * AddPlaceholder(Placeholder<DTYPE> *pPlaceholder);
    Operator<DTYPE>    * AddOperator(Operator<DTYPE> *pOperator);
    Tensorholder<DTYPE>* AddTensorholder(Tensorholder<DTYPE> *pTensorholder);

    Objective<DTYPE>   * SetObjective(Objective<DTYPE> *pObjective);
    Optimizer<DTYPE>   * SetOptimizer(Optimizer<DTYPE> *pOptimizer);

    // =======

    // Optimizer<DTYPE>* SetOptimizer(Optimizer<DTYPE> *pOptimizer);
    int                   FeedData(int numOfTensorholder, ...);

    Operator<DTYPE>     * GetResultOperator();
    Operator<DTYPE>     * GetResult();

    Tensorholder<DTYPE>** GetTensorholder();
    int                   GetTensorholderDegree();

    Objective<DTYPE>    * GetObjective();
    Optimizer<DTYPE>    * GetOptimizer();

    // =======
    float                 GetAccuracy();
    float                 GetLoss();

    // =======
    Operator<DTYPE>     * ForwardPropagate();
    int                   ForwardPropagate(Operator<DTYPE> *pEnd);
    int                   ForwardPropagate(Operator<DTYPE> *pStart, Operator<DTYPE> *pEnd);
    int                   BackPropagate();

    // =======
    int                   Training();
    int                   Testing();

    // =======

    int CreateGraph();

    // temporary
};

#endif  // NEURALNETWORK_H_
