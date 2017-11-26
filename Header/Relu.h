#ifndef RELU_H_
#define RELU_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class Relu : public Operator {
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Relu::Alloc())
    Relu(Operator *pInput, std::string pName) : Operator(pInput, pName) {
        std::cout << "/* Relu::Relu(Operator *) */" << '\n';
        Alloc(pInput);
    }

    virtual ~Relu() {
        std::cout << "Relu::~Relu()" << '\n';
    }

    virtual bool Alloc(Operator *pInput) {
        std::cout << "Relu::Alloc(Operator *, Operator *)" << '\n';

        Tensor *output = new Tensor(GetInputOperator()[0]->GetOutput()->GetShape());
        SetOutput(output);
        Tensor *delta = new Tensor(GetInputOperator()[0]->GetOutput()->GetShape());
        SetDelta(delta);

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int *shape         = GetInputOperator()[0]->GetOutput()->GetShape();
        double *****input  = GetInputOperator()[0]->GetOutput()->GetData();
        double *****output = GetOutput()->GetData();

        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            output[ti][ba][ch][ro][co] = Max(input[ti][ba][ch][ro][co], 0.0);
                        }
                    }
                }
            }
        }

        return true;
    }

    virtual bool ComputeBackPropagate() {
        std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int *shape              = GetOutput()->GetShape();
        double *****output      = GetOutput()->GetData();
        double *****delta       = GetDelta()->GetData();
        double *****delta_input = GetInputOperator()[0]->GetDelta()->GetData();

        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            if (output[ti][ba][ch][ro][co] > 0.0) {
                                delta_input[ti][ba][ch][ro][co] = delta[ti][ba][ch][ro][co];
                            } else {
                                delta_input[ti][ba][ch][ro][co] = 0;
                            }
                        }
                    }
                }
            }
        }

        GetInputOperator()[0]->GetDelta()->PrintData();

        GetDelta()->Reset();

        return true;
    }

    // for relu
    double Max(double data1, double data2) {
        if (data1 >= data2) return data1;
        else return data2;
    }
};

#endif  // RELU_H_
