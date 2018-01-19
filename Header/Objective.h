#ifndef Objective_H_
#define Objective_H_

// #include "MetaParameter.h"
// #include "Optimizer//GradientDescentOptimizer.h"
#include "Operator.h"

template<typename DTYPE> class Objective : public Operator<DTYPE>{
private:
    Tensor<DTYPE> *m_aResult;

    Operator<DTYPE> **m_apInput;

    int m_InputDegree;
    int m_currentInputDegree;

    std::string m_name;

public:
    Objective(std::string pName = "NO NAME");
    Objective(Operator<DTYPE> *pInput, std::string pName = "NO NAME");
    Objective(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName = "NO NAME");

    virtual ~Objective();

    virtual int       Alloc(int numInput, ...);
    virtual void      Delete();

    void              SetResult(Tensor<DTYPE> *pTensor);

    void              IncreaseCurrentOutputDegree();
    void              IncreaseCurrentInputDegree();

    int               _AddInputEdge(Operator<DTYPE> *pInput);
    void              AddEdgebetweenObjectives(Operator<DTYPE> *pInput);

    Tensor<DTYPE>   * GetResult() const;
    Operator<DTYPE>** GetInput() const;
    int               GetInputDegree() const;
    int               GetCurrentInputDegree() const;
    std::string       GetName() const;

    // For Propagate
    int               ForwardPropagate();
    virtual int       ComputeForwardPropagate();

    // For BackPropagate
    int               BackPropagate();
    virtual int       ComputeBackPropagate();
};

#endif  // Objective_H_
