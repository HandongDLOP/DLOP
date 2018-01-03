#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Operator//Placeholder.h"
#include "Operator//Tensorholder.h"

#include "Operator//Reshape.h"

#include "Operator//Relu.h"
#include "Operator//Sigmoid.h"

#include "Operator//Addfc.h"
#include "Operator//Addconv.h"
#include "Operator//MatMulfc.h"
#include "Operator//Convolution.h"
#include "Operator//Maxpooling.h"

#include "Objective//MSE.h"
#include "Objective//SoftmaxCrossEntropy.h"

#include "Optimizer//GradientDescentOptimizer.h"

template<typename DTYPE>
class NeuralNetwork {
private:
    Placeholder<DTYPE> **m_aaPlaceholder;
    Operator<DTYPE> **m_aaOperator;
    Tensorholder<DTYPE> **m_aaTensorholder;
    Optimizer<DTYPE> *m_aOptimizer;

    int numOfPlaceholder;
    int numOfOperator;
    int numOfTensorholder;

public:
    NeuralNetwork();
    virtual ~NeuralNetwork();


    int  Alloc();
    void Delete();

    // =======

    Operator<DTYPE>* AddPlaceholder(Placeholder<DTYPE> *pPlaceholder);
    Operator<DTYPE>* AddOperator(Operator<DTYPE> *pOperator);
    Operator<DTYPE>* AddTensorholder(Tensorholder<DTYPE> *pTensorholder);
    Operator<DTYPE>* AddOptimizer(Optimizer<DTYPE> *pOptimizer);

    // =======

    int Run(Operator<DTYPE> *pStart, Operator<DTYPE> *pEnd);
    int Run(Operator<DTYPE> *pEnd);
    int Run(Optimizer<DTYPE> *pOptimizer);

    // =======

    int ForwardPropagate();
    int BackPropagate();

    // =======

    int CreateGraph();
};

#endif  // NEURALNETWORK_H_
