#ifndef __LAYER__
#define __LAYER__    value

#include "Optimizer_utils.h"

template<typename DTYPE> class Layer : public Operator<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_aaOperator;
    int m_numOfOperator;

private:
    int  Alloc();
    void Delete();

public:
    Layer(std::string pName = "No Name");
    virtual ~Layer();

    Operator<DTYPE>             * AddOperator(Operator<DTYPE> *pOperator);

    Container<Operator<DTYPE> *>* GetOperatorContainer();

    int                           GetNumOfOperator();

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

    Operator<DTYPE>             * GetLastOperator();

    Operator<DTYPE>             * PopOperator();

    int                           ForwardPropagate(int pTime = 0, int pThreadNum = 0);
    int                           BackPropagate(int pTime = 0, int pThreadNum = 0);

    int                           ResetResult();
    int                           ResetGradient();

    void                          PrintInformation();

    void                          SetDeviceCPU();
    void                          SetDeviceCPU(int pnumOfThread);
#ifdef __CUDNN__
    void                          SetDeviceGPU();
    void                          SetDeviceGPU(cudnnHandle_t& pCudnnHandle);

    int                           ForwardPropagateOnGPU(int pTime = 0);
    int                           BackPropagateOnGPU(int pTime = 0);
#endif  // if __CUDNN__
};

#endif  // ifndef __LAYER__
