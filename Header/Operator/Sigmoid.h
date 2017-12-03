#ifndef SIGMOID_H_
#define SIGMOID_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Sigmoid : public Operator<DTYPE> {
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Sigmoid::Alloc())
    Sigmoid(Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pInput, pName) {
        std::cout << "Sigmoid::Sigmoid(Operator *)" << '\n';
        this->Alloc(pInput);
    }

    virtual ~Sigmoid() {
        std::cout << "Sigmoid::~Sigmoid()" << '\n';
    }

    virtual bool Alloc(Operator<DTYPE> *pInput) {
        std::cout << "Sigmoid::Alloc(Operator *, Operator *)" << '\n';

        Tensor<DTYPE> *output = new Tensor<DTYPE>(this->GetInputOperator()[0]->GetOutput()->GetShape());
        this->SetOutput(output);
        Tensor<DTYPE> *delta = new Tensor<DTYPE>(this->GetInputOperator()[0]->GetOutput()->GetShape());
        this->SetDelta(delta);

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int *shape         = this->GetInputOperator()[0]->GetOutput()->GetShape();
        DTYPE *****input  = this->GetInputOperator()[0]->GetOutput()->GetData();
        DTYPE *****output = this->GetOutput()->GetData();

        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            output[ti][ba][ch][ro][co] = this->sigmoid(input[ti][ba][ch][ro][co]);
                        }
                    }
                }
            }
        }


        return true;
    }

    virtual bool ComputeBackPropagate() {
        // std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int *shape              = this->GetOutput()->GetShape();
        DTYPE *****output      = this->GetOutput()->GetData();
        DTYPE *****delta       = this->GetDelta()->GetData();
        this->GetInputOperator()[0]->GetDelta()->Reset();
        DTYPE *****delta_input = this->GetInputOperator()[0]->GetDelta()->GetData();

        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            delta_input[ti][ba][ch][ro][co] = delta[ti][ba][ch][ro][co]
                                                              * output[ti][ba][ch][ro][co]
                                                              * (1 - output[ti][ba][ch][ro][co]);
                        }
                    }
                }
            }
        }

        // GetInputOperator()[0]->GetDelta()->PrintData();

        // GetDelta()->Reset();

        return true;
    }

    // for Sigmoid
    DTYPE sigmoid(DTYPE data) {
        return 1.F / (1.F + (DTYPE)exp(-data));
    }
};

#endif  // SIGMOID_H_
