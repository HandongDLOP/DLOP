#ifndef Objective_H_
#define Objective_H_

#include "Operator//Placeholder.h"
#include "Operator//Tensorholder.h"

#include "Operator//Reshape.h"

#include "Operator//Relu.h"
#include "Operator//Sigmoid.h"

#include "Operator//Add.h"
#include "Operator//Addconv.h"
#include "Operator//MatMul.h"
#include "Operator//Convolution.h"
#include "Operator//Maxpooling.h"
#include "Operator//BatchNormalize.h"
// #include "Operator//DenseBlock.h"

template<typename DTYPE> class Objective {
private:
    Tensor<DTYPE> *m_aResult;
    Tensor<DTYPE> *m_aGradient;

    Operator<DTYPE> *m_pInputOperator;
    Tensor<DTYPE>   *m_pInputTensor;

    Operator<DTYPE> *m_pLabel;

    std::string m_name;

public:
    Objective(std::string pName = "NO NAME");
    Objective(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName = "NO NAME");

    virtual ~Objective();

    virtual int            Alloc(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel);
    virtual void           Delete();

    void                   SetResult(Tensor<DTYPE> *pTensor);
    void                   SetGradient(Tensor<DTYPE> *pTensor);

    Tensor<DTYPE>*         GetResult() const;
    Tensor<DTYPE>*         GetGradient() const;
    Operator<DTYPE>*       GetOperator() const;
    Tensor<DTYPE>*         GetTensor() const;
    Operator<DTYPE>*       GetLabel() const;
    std::string            GetName() const;

    // For Propagate
    virtual Tensor<DTYPE>* ForwardPropagate();

    // For BackPropagate
    virtual Tensor<DTYPE>* BackPropagate();

    DTYPE& operator[](unsigned int index);
};

#endif  // Objective_H_
