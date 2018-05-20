#ifndef __LAYER__
#define __LAYER__    value

#include "Optimizer_utils.h"

template<typename DTYPE> class Layer : public Operator<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_aaOperator;
    Container<Tensorholder<DTYPE> *> *m_aaParameter;

    int m_numOfOperator;
    int m_numOfParameter;

    Device m_Device;

    int m_numOfThread;

public:
    Layer(std::string pName = "No Name");
    virtual ~Layer();

    int  Alloc();
    void Delete();

    // =======

    // Operator<DTYPE>    * AddLayer(Layer<DTYPE> *pLayer);
    Operator<DTYPE>    * AddOperator(Operator<DTYPE> *pOperator);
    Tensorholder<DTYPE>* AddParameter(Tensorholder<DTYPE> *pParameter);

    // =======

    Container<Operator<DTYPE> *>    * GetOperatorContainer();
    Container<Tensorholder<DTYPE> *>* GetParameterContainer();
    int                               GetNumOfOperator();
    int                               GetNumOfParameter();

    Operator<DTYPE>                 * PopOperator();
    Tensorholder<DTYPE>             * PopParameter();

    Tensor<DTYPE>                   * GetResult() const;
    Container<Tensor<DTYPE> *>      * GetResultContainer();

    Tensor<DTYPE>                   * GetGradient() const;
    Container<Tensor<DTYPE> *>      * GetGradientContainer();

    Tensor<DTYPE>                   * GetDelta() const;
    Container<Tensor<DTYPE> *>      * GetDeltaContainer();

    int                               ForwardPropagate(int pThreadNum = 0);
    int                               BackPropagate(int pThreadNum = 0);

#if __CUDNN__
    int                               ForwardPropagateOnGPU(int pTime = 0);
    int                               BackPropagateOnGPU(int pTime = 0);
#endif  // __CUDNN__

    Operator<DTYPE>                 * GetLastOperator();

    void                              SetDeviceCPU();
    void                              SetDeviceCPU(int pnumOfThread);
#if __CUDNN__
    void                              SetDeviceGPU();
    void                              SetCudnnHandle(cudnnHandle_t& pCudnnHandle);
#endif  // if __CUDNN__

    Device                            GetDevice() {
        return m_Device;
    }

    int GetNumOfThread() {
        return m_numOfThread;
    }

    int  ResetResult();
    int  ResetGradient();

    void PrintInformation();
};

#endif  // ifndef __LAYER__
