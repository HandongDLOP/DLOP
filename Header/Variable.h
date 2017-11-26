#ifndef VARIABLE_H_
#define VARIABLE_H_    value

#include <iostream>

#include "Tensor.h"
#include "Operator.h"

class Variable : public Operator {
private:
public:
    Variable(std::string pName) : Operator(pName) {
        std::cout << "Variable::Variable(std::string)" << '\n';
    }

    Variable(Tensor *pTensor, std::string pName, int pTrainable = 0) : Operator(pTensor, pName) {
        std::cout << "Variable::Variable(Tensor *, std::string)" << '\n';

        Alloc(pTensor, pTrainable);
    }

    virtual ~Variable() {
        std::cout << "Variable::~Variable()" << '\n';
    }

    virtual bool Alloc(Tensor *pTensor, int pTrainable) {
        if (pTensor->GetShape()[0] != 1) {
            std::cout << "data has unvalid time dimension" << '\n';
            exit(0);
        }

        SetOutput(pTensor);

        Tensor *gradient = new Tensor(pTensor->GetShape());

        SetGradient(gradient);

        Tensor *delta = new Tensor(pTensor->GetShape());

        SetDelta(delta);

        SetTrainable(pTrainable);

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        return true;
    }

    virtual bool ComputeBackPropagate() {
        std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int *shape        = GetOutput()->GetShape();
        double *****delta = GetDelta()->GetData();
        double *****grad  = GetGradient()->GetData();

        // 이전에 구해져 있던 gradient와 합치기
        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            grad[ti][ba][ch][ro][co] += delta[ti][ba][ch][ro][co];
                        }
                    }
                }
            }
        }

        // Training
        // GetOutput()->PrintData();
        // GetGradient()->PrintData();
        // GetOptimizer()->UpdateWeight(GetOutput(), GetGradient());
        // GetOutput()->PrintData();
        // GetGradient()->PrintData();

        GetDelta()->Reset();
        return true;
    }
};

#endif  // VARIABLE_H_
