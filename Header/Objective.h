#ifndef Objective_H_
#define Objective_H_

#include "NeuralNetwork.h"

template<typename DTYPE> class Objective {
private:
    Tensor<DTYPE> *m_aResult;
    Tensor<DTYPE> *m_aGradient;

    NeuralNetwork<DTYPE> *m_pInputNeuralNetwork;
    Operator<DTYPE> *m_pInputOperator;
    Tensor<DTYPE>   *m_pInputTensor;

    std::string m_name;

public:
    Objective(std::string pName = "NO NAME");
    Objective(NeuralNetwork<DTYPE> *pNeuralNetwork, std::string pName = "NO NAME");

    virtual ~Objective();

    virtual int            Alloc(NeuralNetwork<DTYPE> *pNeuralNetwork);
    virtual void           Delete();

    void                   SetResult(Tensor<DTYPE> *pTensor);
    void                   SetGradient(Tensor<DTYPE> *pTensor);

    Tensor<DTYPE>*         GetResult() const;
    Tensor<DTYPE>*         GetGradient() const;
    NeuralNetwork<DTYPE>*  GetNeuralNetwork() const;
    Operator<DTYPE>*       GetOperator() const;
    Tensor<DTYPE>*         GetTensor() const;
    std::string            GetName() const;

    // For Propagate
    virtual Tensor<DTYPE>* ForwardPropagate(Operator<DTYPE> *pLabel);

    // For BackPropagate
    virtual Tensor<DTYPE>* BackPropagate();
};

#endif  // Objective_H_
