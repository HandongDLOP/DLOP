#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Layer_utils.h"

typedef struct {
    void *m_NN;
    int   m_threadNum;
} ThreadInfo;

template<typename DTYPE> class NeuralNetwork {
private:
    Container<Operator<DTYPE> *> *m_aaOperator;
    Container<Operator<DTYPE> *> *m_aaInput;
    Container<Operator<DTYPE> *> *m_aaParameter;

    int m_OperatorDegree;
    int m_ParameterDegree;

    // 중간에 Loss Function이나 Optimizer가 바뀌는 상황 생각해두기
    LossFunction<DTYPE> *m_aLossFunction;
    Optimizer<DTYPE> *m_aOptimizer;

    Device m_Device;
    int m_numOfThread;

#ifdef __CUDNN__
    cudnnHandle_t m_cudnnHandle;
#endif  // if __CUDNN__

private:
    int  Alloc();
    void Delete();

#ifdef __CUDNN__
    int  AllocOnGPU();
    void DeleteOnGPU();
#endif  // if __CUDNN__

public:
    NeuralNetwork();
    virtual ~NeuralNetwork();


    Operator<DTYPE>    * SetInput(Operator<DTYPE> *pInput);
    Operator<DTYPE>    * AnalyseGraph(Operator<DTYPE> *pResultOperator);

    Operator<DTYPE>    * AddOperator(Operator<DTYPE> *pOperator);
    Operator<DTYPE>    * AddParameter(Operator<DTYPE> *pParameter);

    LossFunction<DTYPE>* SetLossFunction(LossFunction<DTYPE> *pLossFunction);
    Optimizer<DTYPE>   * SetOptimizer(Optimizer<DTYPE> *pOptimizer);

    // =======

    Operator<DTYPE>             * GetResultOperator();
    Operator<DTYPE>             * GetResult();
    Container<Operator<DTYPE> *>* GetOperatorContainer();
    Container<Operator<DTYPE> *>* GetParameter();
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
