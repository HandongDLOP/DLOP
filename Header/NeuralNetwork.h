#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Layer_utils.h"

typedef struct {
    void *m_NN;
    int   m_threadNum;
} ThreadInfo;

template<typename DTYPE> class NeuralNetwork {
private:
#ifdef __CUDNN__
    cudnnHandle_t m_cudnnHandle;
#endif  // if __CUDNN__
    Container<Operator<DTYPE> *> *m_aaOperator;
    Container<Operator<DTYPE> *> *m_aaParameter;
    Container<Layer<DTYPE> *> *m_aaLayer;
    // Parameter

    int m_OperatorDegree;
    int m_ParameterDegree;

    // 중간에 Loss Function이나 Optimizer가 바뀌는 상황 생각해두기
    LossFunction<DTYPE> *m_aLossFunction;
    Optimizer<DTYPE> *m_aOptimizer;

    Device m_Device;
    int m_numOfThread;

public:
    NeuralNetwork();
    virtual ~NeuralNetwork();

    int                  Alloc();
    void                 Delete();

    Operator<DTYPE>    * AddOperator(Operator<DTYPE> *pOperator);
    Operator<DTYPE>    * AddParameter(Operator<DTYPE> *pParameter);
    // Operator<DTYPE>    * AddParameter(Operator<DTYPE> *pParameter);

    LossFunction<DTYPE>* SetLossFunction(LossFunction<DTYPE> *pLossFunction);
    Optimizer<DTYPE>   * SetOptimizer(Optimizer<DTYPE> *pOptimizer);

    // =======

    Operator<DTYPE>             * GetResultOperator();
    Operator<DTYPE>             * GetResult();
    Container<Operator<DTYPE> *>* GetOperatorContainer();
    Container<Operator<DTYPE> *>* GetParameter();
    // Container<Operator<DTYPE> *>* GetParameter();
    LossFunction<DTYPE>         * GetLossFunction();
    Optimizer<DTYPE>            * GetOptimizer();
    float                         GetAccuracy();
    int                           GetMaxIndex(Tensor<DTYPE> *data, int ba, int numOfClass);
    float                         GetLoss();

    int                           ForwardPropagate(int pTime = 0);
    int                           BackPropagate(int pTime = 0);
    static void                 * ForwardPropagateForThread(void *param);
    static void                 * BackPropagateForThread(void *param);

#ifdef __CUDNN__
    int                           ForwardPropagateOnGPU(int pTime = 0);
    int                           BackPropagateOnGPU(int pTime = 0);

    void                          SetDeviceGPU();
#endif  // __CUDNN__

    int                           Training();
    int                           Testing();

    int                           TrainingOnCPU();
    int                           TestingOnCPU();

    int                           TrainingOnMultiThread();     // Multi Threading
    int                           TestingOnMultiThread();      // Multi Threading

    int                           TrainingOnGPU();
    int                           TestingOnGPU();


    void                          SetModeTraining();
    void                          SetModeAccumulating();
    void                          SetModeInferencing();

    void                          SetDeviceCPU();
    void                          SetDeviceCPU(int pNumOfThread);

    // =======
    int                           CreateGraph();
    void                          PrintGraphInformation();

    int                           ResetOperatorResult();
    int                           ResetOperatorGradient();

    int                           ResetLossFunctionResult();
    int                           ResetLossFunctionGradient();

    int                           ResetParameterGradient();

    Operator<DTYPE>             * SerchOperator(std::string pName);
};

#endif  // NEURALNETWORK_H_
