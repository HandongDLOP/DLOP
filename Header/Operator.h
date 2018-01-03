#ifndef OPERATOR_H_
#define OPERATOR_H_

// #include "MetaParameter.h"
// #include "Optimizer//GradientDescentOptimizer.h"
#include "Tensor.h"

template<typename DTYPE> class Operator {
private:
    Tensor<DTYPE> *m_aResult;
    Tensor<DTYPE> *m_aGradient;
    Tensor<DTYPE> *m_aDelta;

    Operator<DTYPE> **m_apOutput;
    Operator<DTYPE> **m_apInput;

    int m_OutputDegree;
    int m_InputDegree;
    int m_currentOutputDegree;
    int m_currentInputDegree;

    std::string m_name;

public:
    Operator(std::string pName = "NO NAME");
    Operator(Operator<DTYPE> *pInput, std::string pName = "NO NAME");
    Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName = "NO NAME");
    virtual ~Operator();

    virtual int       Alloc(int numInput, ...);
    virtual void      Delete();

    void              SetResult(Tensor<DTYPE> *pTensor);
    void              SetGradient(Tensor<DTYPE> *pTensor);
    void              SetDelta(Tensor<DTYPE> *pTensor);

    void              IncreaseCurrentOutputDegree();
    void              IncreaseCurrentInputDegree();

    int               _AddInputEdge(Operator<DTYPE> *pInput);
    int               _AddOutputEdge(Operator<DTYPE> *pOutput);
    void              AddEdgebetweenOperators(Operator<DTYPE> *pInput);

    Tensor<DTYPE>   * GetResult() const;
    Tensor<DTYPE>   * GetGradient() const;
    Tensor<DTYPE>   * GetDelta() const;
    Operator<DTYPE>** GetInput() const;
    Operator<DTYPE>** GetOutput() const;
    int               GetOutputDegree() const;
    int               GetInputDegree() const;
    int               GetCurrentOutputDegree() const;
    int               GetCurrentInputDegree() const;
    std::string       GetName() const;

    // For Propagate
    int               ForwardPropagate();
    virtual int       ComputeForwardPropagate();

    // For BackPropagate
    int               BackPropagate();
    virtual int       ComputeBackPropagate();
};

#endif  // OPERATOR_H_
