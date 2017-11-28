#ifndef SOFTMAX_CROSS_ENTROPY_H_
#define SOFTMAX_CROSS_ENTROPY_H_    value

#include <iostream>
#include <string>
#include <math.h>

#include "Tensor.h"
#include "Operator.h"

class Softmax_Cross_Entropy : public Operator {
private:
    Tensor * m_aSoftmax_Result = NULL;

public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Softmax_Cross_Entropy::Alloc())
    Softmax_Cross_Entropy(Operator *pInput1, Operator *pInput2) : Operator(pInput1, pInput2) {
        std::cout << "Softmax_Cross_Entropy::Softmax_Cross_Entropy(Operator *, MetaParameter *)" << '\n';
        Alloc(pInput1, pInput2);
    }

    Softmax_Cross_Entropy(Operator *pInput1, Operator *pInput2, std::string pName) : Operator(pInput1, pInput2, pName) {
        std::cout << "Softmax_Cross_Entropy::Softmax_Cross_Entropy(Operator *, MetaParameter *, std::string)" << '\n';
        Alloc(pInput1, pInput2);
    }

    virtual ~Softmax_Cross_Entropy() {
        std::cout << "Softmax_Cross_Entropy::~Softmax_Cross_Entropy()" << '\n';

        delete m_aSoftmax_Result;
    }

    virtual bool Alloc(Operator *pInput1, Operator *pInput2) {
        std::cout << "Softmax_Cross_Entropy::Alloc(Operator *, Operator *)" << '\n';
        // if pInput1 and pInput2의 shape가 다르면 abort

        int *shape     = GetInputOperator()[0]->GetOutput()->GetShape();
        Tensor *output = new Tensor(shape[0], shape[1], 1, 1, 1);
        SetOutput(output);

        m_aSoftmax_Result = new Tensor(shape);

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int *shape         = GetInputOperator()[0]->GetOutput()->GetShape();
        double *****input0 = GetInputOperator()[0]->GetOutput()->GetData();
        double *****input1 = GetInputOperator()[1]->GetOutput()->GetData();

        GetOutput()->Reset();
        double *****output         = GetOutput()->GetData();
        double *****softmax_result = m_aSoftmax_Result->GetData();

        int Time    = shape[0];
        int Batch   = shape[1];
        int Channel = shape[2];
        int Row     = shape[3];
        int Col     = shape[4];

        double sum[Time][Batch] = { 0.0 };
        int    num_of_output    = Channel * Row * Col;

        for (int ti = 0; ti < Time; ti++) {
            for (int ba = 0; ba < Batch; ba++) {
                for (int ch = 0; ch < Channel; ch++) {
                    for (int ro = 0; ro < Row; ro++) {
                        for (int co = 0; co < Col; co++) {
                            sum[ti][ba] += input0[ti][ba][ch][ro][co];
                        }
                    }
                }
            }
        }

        for (int ti = 0; ti < Time; ti++) {
            for (int ba = 0; ba < Batch; ba++) {
                for (int ch = 0; ch < Channel; ch++) {
                    for (int ro = 0; ro < Row; ro++) {
                        for (int co = 0; co < Col; co++) {
                            softmax_result[ti][ba][ch][ro][co] = input0[ti][ba][ch][ro][co] / sum[ti][ba];
                        }
                    }
                }
            }
        }

        for (int ti = 0; ti < Time; ti++) {
            for (int ba = 0; ba < Batch; ba++) {
                for (int ch = 0; ch < Channel; ch++) {
                    for (int ro = 0; ro < Row; ro++) {
                        for (int co = 0; co < Col; co++) {
                            output[ti][ba][0][0][0] = cross_entropy(input1[ti][ba][ch][ro][co],
                                                                    softmax_result[ti][ba][ch][ro][co],
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

        double *****input1       = GetInputOperator()[1]->GetOutput()->GetData();
        double *****softmax_result = m_aSoftmax_Result->GetData();
        double *****delta_Input0 = GetInputOperator()[0]->GetDelta()->GetData();

        int Time    = shape[0];
        int Batch   = shape[1];
        int Channel = shape[2];
        int Row     = shape[3];
        int Col     = shape[4];

        for (int ti = 0; ti < Time; ti++) {
            for (int ba = 0; ba < Batch; ba++) {
                for (int ch = 0; ch < Channel; ch++) {
                    for (int ro = 0; ro < Row; ro++) {
                        for (int co = 0; co < Col; co++) {
                            delta_Input0[ti][ba][ch][ro][co] = input1[ti][ba][ch][ro][co] - softmax_result[ti][ba][ch][ro][co];
                        }
                    }
                }
            }
        }

        // GetInputOperator()[0]->GetDelta()->PrintData();

        return true;
    }

    double cross_entropy(double label, double prediction, int num_of_output) {
        return -label *log(prediction) / num_of_output;
    }
};

#endif  // SOFTMAX_CROSS_ENTROPY_H_
