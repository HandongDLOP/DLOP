#ifndef __LAYER__
#define __LAYER__    value

#include "Optimizer_utils.h"

template<typename DTYPE> class Layer : public Operator<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_aaOperator;
    Container<Operator<DTYPE> *> *m_aaParameter;

    int m_numOfOperator;
    int m_numOfParameter;

    int  Alloc();
    void Delete();

public:
    Layer(std::string pName = "No Name");
    virtual ~Layer();

    Operator<DTYPE>             * AddOperator(Operator<DTYPE> *pOperator);
    Operator<DTYPE>             * AddParameter(Operator<DTYPE> *pParameter);


    Container<Operator<DTYPE> *>* GetOperatorContainer();
    Container<Operator<DTYPE> *>* GetParameterContainer();

    int                           GetNumOfOperator();
    int                           GetNumOfParameter();

    Operator<DTYPE>            ** GetOutput();
    Container<Operator<DTYPE> *>* GetOutputContainer();
    Operator<DTYPE>            ** GetInput();
    Container<Operator<DTYPE> *>* GetInputContainer();

    Tensor<DTYPE>               * GetResult() const;
    Container<Tensor<DTYPE> *>  * GetResultContainer();

    Tensor<DTYPE>               * GetGradient() const;
    Container<Tensor<DTYPE> *>  * GetGradientContainer();

    Tensor<DTYPE>               * GetDelta() const;
    Container<Tensor<DTYPE> *>  * GetDeltaContainer();

    Operator<DTYPE>             * PopOperator();
    Operator<DTYPE>             * PopParameter();

    int                           ForwardPropagate(int pTime = 0, int pThreadNum = 0);
    int                           BackPropagate(int pTime = 0, int pThreadNum = 0);

    Operator<DTYPE>             * GetLastOperator();

    void                          SetDeviceCPU();
    void                          SetDeviceCPU(int pnumOfThread);
#ifdef __CUDNN__
    int                           ForwardPropagateOnGPU(int pTime = 0);
    int                           BackPropagateOnGPU(int pTime = 0);
    void                          SetDeviceGPU();
    void                          SetDeviceGPU(cudnnHandle_t& pCudnnHandle);
#endif  // if __CUDNN__

    int                           ResetResult();
    int                           ResetGradient();

    void                          PrintInformation();
};

#endif  // ifndef __LAYER__
