#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Layer_utils.h"

typedef struct {
    void *m_NN;
    int   m_time;
    int   m_threadNum;
} ThreadInfo;

template<typename DTYPE> class NeuralNetwork {
private:
#if __CUDNN__
    cudnnHandle_t m_cudnnHandle;
#endif  // if __CUDNN__
    Container<Operator<DTYPE> *> *m_aaOperator;
    Container<Tensorholder<DTYPE> *> *m_aaTensorholder;
    Container<Layer<DTYPE> *> *m_aaLayer;
    // Parameter

    int m_OperatorDegree;
    int m_TensorholderDegree;

    Objective<DTYPE> *m_aObjective;
    Optimizer<DTYPE> *m_aOptimizer;

    Device m_Device;
    int m_numOfThread;

public:
    NeuralNetwork();
    virtual ~NeuralNetwork();

    int  Alloc();
    void Delete();

    // =======

    // 추후 직접 변수를 만들지 않은 operator* + operator*의 변환 변수도 자동으로 할당될 수 있도록 Operator와 NN class를 수정해야 한다.
    Operator<DTYPE>    * AddOperator(Operator<DTYPE> *pOperator);
    Tensorholder<DTYPE>* AddTensorholder(Tensorholder<DTYPE> *pTensorholder);
    Tensorholder<DTYPE>* AddParameter(Tensorholder<DTYPE> *pTensorholder);
    // Operator<DTYPE>    * AddLayer(Layer<DTYPE> *pLayer);

    Objective<DTYPE>   * SetObjective(Objective<DTYPE> *pObjective);
    Optimizer<DTYPE>   * SetOptimizer(Optimizer<DTYPE> *pOptimizer);

    // =======

    Operator<DTYPE>                 * GetResultOperator();
    Operator<DTYPE>                 * GetResult();
    Container<Operator<DTYPE> *>    * GetOperatorContainer();


    Container<Tensorholder<DTYPE> *>* GetTensorholder();
    Container<Tensorholder<DTYPE> *>* GetParameter();

    Objective<DTYPE>                * GetObjective();
    Optimizer<DTYPE>                * GetOptimizer();

    // =======
    float                             GetAccuracy();
    int                               GetMaxIndex(Tensor<DTYPE> *data, int ba, int numOfClass);
    float                             GetLoss();

    // =======
    int                               ForwardPropagate();
    int                               BackPropagate();
    static void                     * ForwardPropagate_T(void *param);
    static void                     * BackPropagate_T(void *param);

    // =======
    int                               Training();
    int                               Testing();

    int                               _Training_MT();
    int                               _Testing_MT();

    // ============
    void                              SetModeTraining();
    void                              SetModeAccumulating();
    void                              SetModeInferencing();

#if __CUDNN__
    void                              SetDeviceGPU();
#endif  // __CUDNN__

    void                              SetDeviceCPU();
    void                              SetDeviceCPU(int pNumOfThread);

    // =======
    int                               CreateGraph();
    void                              PrintGraphInformation();

    // reset value
    int                               ResetOperatorResult();
    int                               ResetOperatorGradient();

    int                               ResetObjectiveResult();
    int                               ResetObjectiveGradient();

    int                               ResetParameterGradient();

    // debug
    Operator<DTYPE>                 * SerchOperator(std::string pName);
};

#endif  // NEURALNETWORK_H_
