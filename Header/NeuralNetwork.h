#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

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

#include "Objective//MSE.h"
#include "Objective//SoftmaxCrossEntropy.h"

#include "Optimizer//GradientDescentOptimizer.h"

template<typename DTYPE> class NeuralNetwork {
private:
    Placeholder<DTYPE> **m_aaPlaceholder;
    Operator<DTYPE> **m_aaOperator;
    Tensorholder<DTYPE> **m_aaTensorholder;

    int m_PlaceholderDegree;
    int m_OperatorDegree;
    int m_TensorholderDegree;

    Objective<DTYPE> *m_aObjectiveFunction;

    Optimizer<DTYPE> *m_aOptimizer;

public:
    NeuralNetwork();
    virtual ~NeuralNetwork();

    // int  Alloc();
    void Delete();

    // =======

    // 추후 직접 변수를 만들지 않은 operator* + operator*의 변환 변수도 자동으로 할당될 수 있도록 Operator와 NN class를 수정해야 한다.
    Placeholder<DTYPE> * AddPlaceholder(Placeholder<DTYPE> *pPlaceholder);
    Operator<DTYPE>    * AddOperator(Operator<DTYPE> *pOperator);
    Tensorholder<DTYPE>* AddTensorholder(Tensorholder<DTYPE> *pTensorholder);

    // =======

    Objective<DTYPE>* SetObjectiveFunction(Objective<DTYPE> *pObjectiveFunction);
    Optimizer<DTYPE>* SetOptimizer(Optimizer<DTYPE> *pOptimizer);
    int FeedData(int numOfTensorholder, ...);

    // =======

    Operator<DTYPE>* Training();
    Operator<DTYPE>* Training(Operator<DTYPE> *pEnd);
    Operator<DTYPE>* Testing();
    Operator<DTYPE>* Testing(Operator<DTYPE> *pEnd);
    Operator<DTYPE>* Testing(Operator<DTYPE> *pStart, Operator<DTYPE> *pEnd);

    // =======
    int ForwardPropagate();
    int ForwardPropagate(Operator<DTYPE> *pEnd);
    int ForwardPropagate(Operator<DTYPE> *pStart, Operator<DTYPE> *pEnd);
    int BackPropagate();

    // =======

    int CreateGraph();

    // temporary
};

#endif  // NEURALNETWORK_H_
