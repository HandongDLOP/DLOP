#ifndef VARIABLE_H_
#define VARIABLE_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class Variable : public Operator {
public:
    Variable(std::string pName) : Operator(pName) {
        std::cout << "Variable::Variable(std::string)" << '\n';
    }

    Variable(Tensor *pTensor, std::string pName) : Operator(pTensor, pName) {
        std::cout << "Variable::Variable(Tensor *, std::string)" << '\n';

        Alloc(pTensor);
    }

    virtual ~Variable() {
        std::cout << "Variable::~Variable()" << '\n';
    }

    virtual bool Alloc(Tensor *pTensor) {
        SetOutputDim(pTensor->Getshape());

        SetOutput(pTensor);

        Tensor *temp_Gradient = new Tensor(pTensor->Getshape());

        SetGradient(temp_Gradient);

        Tensor *temp_delta = new Tensor(pTensor->Getshape());

        SetDelta(temp_delta);

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        return true;
    }

    virtual bool ComputeBackPropagate() {
        std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int size = GetOutput()->GetFlatDim();

        float *delta = GetDelta()->GetData();

        float *grad = GetGradient()->GetData();

        float *_grad = new float (size);

        // 이전에 구해져 있던 gradient와 합치기
        for (int i = 0; i < size; i++) {
            _grad[i] = delta[i] + grad[i];
        }

        SetGradient(_grad);

        GetGradient()->PrintData();

        return true;
    }
};

#endif  // VARIABLE_H_
