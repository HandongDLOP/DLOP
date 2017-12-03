#ifndef MSE_H_
#define MSE_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class MSE : public Operator<DTYPE>{
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (MSE::Alloc())
    MSE(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1) : Operator<DTYPE>(pInput0, pInput1) {
        std::cout << "MSE::MSE(Operator<DTYPE> *, MetaParameter *)" << '\n';
        this->Alloc(pInput0, pInput1);
    }

    MSE(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        std::cout << "MSE::MSE(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        this->Alloc(pInput0, pInput1);
    }

    virtual ~MSE() {
        std::cout << "MSE::~MSE()" << '\n';
    }

    virtual bool Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1) {
        std::cout << "MSE::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        // if pInput0 and pInput1의 shape가 다르면 abort

        int *shape            = pInput0->GetOutput()->GetShape();
        Tensor<DTYPE> *output = new Tensor<DTYPE>(shape[0], shape[1], 1, 1, 1);
        this->SetOutput(output);

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int *shape        = this->GetInputOperator()[0]->GetOutput()->GetShape();
        DTYPE *****input0 = this->GetInputOperator()[0]->GetOutput()->GetData();
        DTYPE *****input1 = this->GetInputOperator()[1]->GetOutput()->GetData();

        this->GetOutput()->Reset();
        DTYPE *****output = this->GetOutput()->GetData();
        int num_of_output = shape[2] * shape[3] * shape[4];

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

        int *shape         = this->GetInputOperator()[0]->GetOutput()->GetShape();
        int  num_of_output = shape[2] * shape[3] * shape[4];  /* * InputDim0->GetDim()[2] == ch*/;
        DTYPE *****input0  = this->GetInputOperator()[0]->GetOutput()->GetData();
        DTYPE *****input1  = this->GetInputOperator()[1]->GetOutput()->GetData();

        this->GetInputOperator()[0]->GetDelta()->Reset();
        DTYPE *****delta_Input0 = this->GetInputOperator()[0]->GetDelta()->GetData();

        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            delta_Input0[ti][ba][ch][ro][co] = (input0[ti][ba][ch][ro][co] - input1[ti][ba][ch][ro][co]) / num_of_output;
                            // delta_Input0[ti][ba][ch][ro][co] = (input0[ti][ba][ch][ro][co] - input1[ti][ba][ch][ro][co]);
                        }
                    }
                }
            }
        }

        // GetInputOperator()[0]->GetDelta()->PrintData();

        return true;
    }

    DTYPE Error(DTYPE input0, DTYPE input1, int num_of_output) {
        return (input0 - input1) * (input0 - input1) / num_of_output * 0.5;
        // return (input0 - input1) * (input0 - input1) / 2.0;
    }
};

#endif  // MSE_H_
