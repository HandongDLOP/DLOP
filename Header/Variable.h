#ifndef VARIABLE_H_
#define VARIABLE_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class Variable : public Operator {
private:
    int m_isTrainable = 0;

public:
    Variable(std::string pName) : Operator(pName) {
        std::cout << "Variable::Variable(std::string)" << '\n';
    }

    Variable(Tensor *pTensor, std::string pName, int pisTrainable = 0) : Operator(pTensor, pName) {
        std::cout << "Variable::Variable(Tensor *, std::string)" << '\n';

        Alloc(pTensor, pisTrainable);
    }

    virtual ~Variable() {
        std::cout << "Variable::~Variable()" << '\n';
    }

    virtual bool Alloc(Tensor *pTensor, int pisTrainable) {
        SetOutputDim(pTensor->Getshape());

        SetOutput(pTensor);

        Tensor *temp_Gradient = new Tensor(pTensor->Getshape());

        SetGradient(temp_Gradient);

        Tensor *temp_delta = new Tensor(pTensor->Getshape());

        SetDelta(temp_delta);

        m_isTrainable = pisTrainable;

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

        GetOutput()->PrintData();

        if (m_isTrainable == 1) {

            GetGradient()->PrintData();

            Optimizer *optimizer = GetOptimizer();

            optimizer->UpdateWeight(GetOutput(), GetGradient());

            GetOutput()->PrintData();

            GetGradient()->PrintData();
        }

        return true;
    }
};

#endif  // VARIABLE_H_
