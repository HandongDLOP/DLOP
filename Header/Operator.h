#ifndef OPERATOR_H_
#define OPERATOR_H_

#include "Tensor_utils.h"
#include "Container.h"

enum Mode {
    TRAINING,
    ACCUMULATING,
    INFERENCING,
};

template<typename DTYPE> class Operator {
private:
    Container<Operator<DTYPE> *> *m_apOutput;
    Container<Operator<DTYPE> *> *m_apInput;
    Container<Tensor<DTYPE> *> *m_aaResult;
    Container<Tensor<DTYPE> *> *m_aaGradient;
    std::string m_name;
    Device m_Device;
    int m_numOfThread;
    int m_numOfParameter;
    int m_isTensorholder;
    int m_isTrainable;

private:
    int  Alloc();
    int  Alloc(int numInput, ...);
    void Delete();

    int  _AddInputEdge(Operator<DTYPE> *pInput);
    int  _AddOutputEdge(Operator<DTYPE> *pOutput);

#ifdef __CUDNN__


#endif  // __CUDNN__

public:
    Operator(std::string pName = "NO NAME");
    Operator(Operator<DTYPE> *pInput, std::string pName = "NO NAME");
    Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName = "NO NAME");
    virtual ~Operator();

    int                                       AddEdgebetweenOperators(Operator<DTYPE> *pInput);
    int                                       AddEdgebetweenOperators(int numInput, ...);
    int                                       AddResult(Tensor<DTYPE> *pTensor);
    int                                       AddGradient(Tensor<DTYPE> *pTensor);
    int                                       AddDelta(Tensor<DTYPE> *pTensor);
    int                                       SetResult(Tensor<DTYPE> *pTensor); // 0 or 1 일 때만 진행 가능
    int                                       SetGradient(Tensor<DTYPE> *pTensor);
    int                                       SetDelta(Tensor<DTYPE> *pTensor);

    virtual void                              SetModeTraining();
    virtual void                              SetModeAccumulating();
    virtual void                              SetModeInferencing();


    Operator<DTYPE>                        ** GetOutput();
    Container<Operator<DTYPE> *>            * GetOutputContainer();
    Operator<DTYPE>                        ** GetInput();
    Container<Operator<DTYPE> *>            * GetInputContainer();
    virtual Tensor<DTYPE>                   * GetResult() const;
    virtual Container<Tensor<DTYPE> *>      * GetResultContainer();
    virtual Tensor<DTYPE>                   * GetGradient() const;
    virtual Container<Tensor<DTYPE> *>      * GetGradientContainer();
    virtual Tensor<DTYPE>                   * GetDelta() const;
    virtual Container<Tensor<DTYPE> *>      * GetDeltaContainer();

    std::string                               GetName() const;
    virtual Device                            GetDevice();
    int                                       GetNumOfThread();

    virtual int                               ForwardPropagate(int pTime = 0, int pThreadNum = 0);
    virtual int                               BackPropagate(int pTime = 0, int pThreadNum = 0);
    // reset value
    virtual int                               ResetResult();
    virtual int                               ResetGradient();

    virtual void                              SetDeviceCPU();
    virtual void                              SetDeviceCPU(int pNumOfThread);

    virtual Container<Tensorholder<DTYPE> *>* GetParameterContainer();
    virtual int                               GetNumOfParameter();
    virtual Tensorholder<DTYPE>             * PopParameter();

    virtual void                              PrintInformation();

#ifdef __CUDNN__

    cudnnHandle_t m_pCudnnHandle;
    cudnnHandle_t& GetCudnnHandle();
    virtual void   InitializeAttributeForGPU();
    virtual void   SetCudnnHandle(cudnnHandle_t& pCudnnHandle);
    void           cudnnResize(int size, float *data);

    virtual int    ForwardPropagateOnGPU(int pTime = 0);
    virtual int    BackPropagateOnGPU(int pTime = 0);

    virtual int    SetResultOnCPU();
    virtual int    SetGradientOnCPU();

    virtual void   SetDeviceGPU();

    virtual int    SetResultOnGPU();
    virtual int    SetGradientOnGPU();

#endif  // if __CUDNN__
};

#endif  // OPERATOR_H_
