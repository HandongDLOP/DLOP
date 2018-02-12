#ifndef MODEL_H_
#define MODEL_H_

#include "Optimizer//GradientDescentOptimizer.h"
#include "..//Header//Temporary_method.h"

template<typename DTYPE> class Model {
private:
    NeuralNetwork<DTYPE> **m_aaNeuralNetworks;
    Objective<DTYPE>     *m_aObjective;
    Optimizer<DTYPE>     *m_aOptimizer;

    int m_NeuralNetworkDegree;

public:
    Model();
    virtual ~Model();

    int  Alloc();
    void Delete();

    // =======
    NeuralNetwork<DTYPE>* AddNeuralNetwork(NeuralNetwork<DTYPE> *pNeuralNetwork);
    Objective<DTYPE>*     SetObjective(Objective<DTYPE> *pObjective);
    Optimizer<DTYPE>*     SetOptimizer(Optimizer<DTYPE> *pOptimizer);

    // =======
    int                   Training();
    int                   Testing();

    // =======
    NeuralNetwork<DTYPE>** GetNeuralNetworks();
    Objective<DTYPE>*     GetObjective();
    Optimizer<DTYPE>*     GetOptimizer();

    //=======
    float GetAccuracy();
    float GetLoss();

};

#endif  // MODEL_H_
