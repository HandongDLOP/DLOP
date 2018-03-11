#ifndef OPERATOR_H_
#define OPERATOR_H_

#include "Tensor_utils.h"
#include "Container.h"
#define VALID    0
#define SAME     1

template<typename DTYPE> class Operator {
private:
    Container<Tensor<DTYPE> *> *m_aaResult;
    Container<Tensor<DTYPE> *> *m_aaGradient;
    Container<Tensor<DTYPE> *> *m_aaDelta;

    Container<Operator<DTYPE> *> *m_apOutput;
    Container<Operator<DTYPE> *> *m_apInput;

    int m_OutputDegree;
    int m_InputDegree;

    int m_currentOutputDegree;
    int m_currentInputDegree;

    std::string m_name;

public:
#if __CUDNN__
    cudnnHandle_t m_pCudnnHandle;
    cudnnHandle_t& GetCudnnHandle();
    void           SetCudnnHandle(cudnnHandle_t& pCudnnHandle);
    void           cudnnResize(int size, float *data);
#endif  // if __CUDNN__

    Operator(std::string pName = "NO NAME");
    Operator(Operator<DTYPE> *pInput, std::string pName = "NO NAME");
    Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName = "NO NAME");
    virtual ~Operator();

    virtual int                   Alloc();
    virtual int                   Alloc(int numInput, ...);
    virtual void                  Delete();

    void                          SetResult(Tensor<DTYPE> *pTensor);
    void                          AddResult(Tensor<DTYPE> *pTensor);

    void                          SetGradient(Tensor<DTYPE> *pTensor);
    void                          AddGradient(Tensor<DTYPE> *pTensor);

    void                          SetDelta(Tensor<DTYPE> *pTensor);
    void                          AddDelta(Tensor<DTYPE> *pTensor);

    void                          IncreaseCurrentOutputDegree();
    void                          IncreaseCurrentInputDegree();

    int                           _AddInputEdge(Operator<DTYPE> *pInput);
    int                           _AddOutputEdge(Operator<DTYPE> *pOutput);
    void                          AddEdgebetweenOperators(Operator<DTYPE> *pInput);

    Tensor<DTYPE>               * GetResult() const;
    Container<Tensor<DTYPE> *>  * GetResultContainer();

    Tensor<DTYPE>               * GetGradient() const;
    Container<Tensor<DTYPE> *>  * GetGradientContainer();

    Tensor<DTYPE>               * GetDelta() const;
    Container<Tensor<DTYPE> *>  * GetDeltaContainer();

    Operator<DTYPE>            ** GetOutput();
    Container<Operator<DTYPE> *>* GetOutputContatiner();

    Operator<DTYPE>            ** GetInput();
    Container<Operator<DTYPE> *>* GetInputContatiner();

    int                           GetOutputDegree() const;
    int                           GetInputDegree() const;

    int                           GetCurrentOutputDegree() const;
    int                           GetCurrentInputDegree() const;
    std::string                   GetName() const;

    // Operator<DTYPE>             * Concatenate(Operator<DTYPE> *src, Operator<DTYPE> *dst, int axis = 0);

    // For Propagate
    int         ForwardPropagate();
    virtual int ComputeForwardPropagate();

    // For BackPropagate
    int         BackPropagate();
    virtual int ComputeBackPropagate();
};

#endif  // OPERATOR_H_
