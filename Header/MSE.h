#ifndef MSE_H_
#define MSE_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class MSE : public Operator {
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (MSE::Alloc())
    MSE(Operator *pInput1, Operator *pInput2) : Operator(pInput1, pInput2) {
        std::cout << "MSE::MSE(Operator *, MetaParameter *)" << '\n';
        Alloc(pInput1, pInput2);
    }

    MSE(Operator *pInput1, Operator *pInput2, std::string pName) : Operator(pInput1, pInput2, pName) {
        std::cout << "MSE::MSE(Operator *, MetaParameter *, std::string)" << '\n';
        Alloc(pInput1, pInput2);
    }

    virtual ~MSE() {
        std::cout << "MSE::~MSE()" << '\n';
    }

    virtual bool Alloc(Operator *pInput1, Operator *pInput2) {
        std::cout << "MSE::Alloc(Operator *, Operator *)" << '\n';
        // if pInput1 and pInput2의 shape가 다르면 abort

        int *shape     = GetInputOperator()[0]->GetOutput()->GetShape();
        Tensor *output = new Tensor(shape[0], shape[1], 1, 1, 1);
        SetOutput(output);

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int *shape         = GetInputOperator()[0]->GetOutput()->GetShape();
        double *****input0 = GetInputOperator()[0]->GetOutput()->GetData();
        double *****input1 = GetInputOperator()[1]->GetOutput()->GetData();
        GetOutput()->Reset();
        double *****output = GetOutput()->GetData();
        int num_of_output  = shape[2] * shape[3] * shape[4];

        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            output[ti][ba][0][0][0] += Error(input0[ti][ba][ch][ro][co],
                                                             input1[ti][ba][ch][ro][co],
                                                             num_of_output);
                        }
                    }
                }
            }
        }

        // GetInputOperator()[0]->GetOutput()->PrintData();
        // GetInputOperator()[1]->GetOutput()->PrintData();
        // GetOutput()->PrintData();

        return true;
    }

    virtual bool ComputeBackPropagate() {
        // std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int *shape               = GetInputOperator()[0]->GetOutput()->GetShape();
        int  num_of_output       = shape[2] * shape[3] * shape[4];  /* * InputDim0->GetDim()[2] == ch*/;
        double *****input0       = GetInputOperator()[0]->GetOutput()->GetData();
        double *****input1       = GetInputOperator()[1]->GetOutput()->GetData();
        double *****delta_Input0 = GetInputOperator()[0]->GetDelta()->GetData();

        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            delta_Input0[ti][ba][ch][ro][co] = (input0[ti][ba][ch][ro][co] - input1[ti][ba][ch][ro][co]) / num_of_output;
                        }
                    }
                }
            }
        }

        // GetInputOperator()[0]->GetDelta()->PrintData();

        return true;
    }

    double Error(double input0, double input1, int num_of_output) {
        return (input0 - input1) * (input0 - input1) / num_of_output * 0.5;
    }
};

#endif  // MSE_H_
