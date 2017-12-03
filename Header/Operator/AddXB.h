#ifndef ADD_H_
#define ADD_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Add : public Operator<DTYPE> {
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Add::Alloc())
    Add(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1) : Operator<DTYPE>(pInput0, pInput1) {
        std::cout << "Add::Add(Operator<DTYPE> *, MetaParameter *)" << '\n';
        this->Alloc(pInput0, pInput1);
    }

    Add(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        std::cout << "Add::Add(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        this->Alloc(pInput0, pInput1);
    }

    virtual ~Add() {
        std::cout << "Add::~Add()" << '\n';
    }

    virtual bool Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1) {
        std::cout << "Add::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        // if pInput0 and pInput1의 shape가 다르면 abort

        int *shape_Input0 = pInput0->GetOutput()->GetShape();
        int *shape_Input1 = pInput1->GetOutput()->GetShape();

        if (shape_Input0[4] != shape_Input1[4]) {
            std::cout << "data has different col dimension" << '\n';
            exit(0);
        }

        if (shape_Input0[3] != shape_Input1[3]) {
            std::cout << "data has different row dimension" << '\n';
            exit(0);
        }

        if (shape_Input0[2] != shape_Input1[2]) {
            std::cout << "data has different channel dimension" << '\n';
            exit(0);
        }

        // factory method 작업

        if (shape_Input0[1] != shape_Input1[1]) {
            if ((shape_Input0[1] == 1) || (shape_Input1[1] == 1)) {
                if (shape_Input0[1] < shape_Input1[1]) {
                    this->GetInputOperator()[0] = pInput1;
                    this->GetInputOperator()[1] = pInput0;
                }
            } else {
                std::cout << "data has unvalid batch dimension" << '\n';
                exit(0);
            }
        }


        // if (shape_Input0[0] != shape_Input1[0]) {
        // if ((shape_Input0[0] == 1) || (shape_Input1[0] == 1)) {
        // if (shape_Input0[0] < shape_Input1[0]) {
        // std::cout << "data has unvalid batch dimension" << '\n';
        // exit(0);
        // }
        // } else {
        // std::cout << "data has unvalid time dimension" << '\n';
        // exit(0);
        // }
        // }

        // batch가 큰것으로 이미 정렬된 상태
        Tensor<DTYPE> *output = new Tensor<DTYPE>(this->GetInputOperator()[0]->GetOutput()->GetShape());

        this->SetOutput(output);

        Tensor<DTYPE> *delta = new Tensor<DTYPE>(this->GetInputOperator()[0]->GetOutput()->GetShape());

        this->SetDelta(delta);

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int *shape_Input0 = this->GetInputOperator()[0]->GetOutput()->GetShape();
        // int *shape_Input1       = GetInputOperator()[1]->GetOutput()->GetShape();
        DTYPE *****input0 = this->GetInputOperator()[0]->GetOutput()->GetData();
        DTYPE *****input1 = this->GetInputOperator()[1]->GetOutput()->GetData();
        DTYPE *****output = this->GetOutput()->GetData();

        // factory method

        // if ((shape_Input0[0] == shape_Input1[0]) && (shape_Input0[1] == shape_Input1[1])) {
        // for (int ti = 0; ti < shape_Input0[0]; ti++) {
        // for (int ba = 0; ba < shape_Input0[1]; ba++) {
        // for (int ch = 0; ch < shape_Input0[2]; ch++) {
        // for (int ro = 0; ro < shape_Input0[3]; ro++) {
        // for (int co = 0; co < shape_Input0[4]; co++) {
        // output[ti][ba][ch][ro][co] = input0[ti][ba][ch][ro][co] + input1[ti][ba][ch][ro][co];
        // }
        // }
        // }
        // }
        // }
        // } else if ((shape_Input0[1] != shape_Input1[1]) && (shape_Input1[1] == 1)) {
        for (int ti = 0; ti < shape_Input0[0]; ti++) {
            for (int ba = 0; ba < shape_Input0[1]; ba++) {
                for (int ch = 0; ch < shape_Input0[2]; ch++) {
                    for (int ro = 0; ro < shape_Input0[3]; ro++) {
                        for (int co = 0; co < shape_Input0[4]; co++) {
                            output[ti][ba][ch][ro][co] = input0[ti][ba][ch][ro][co] + input1[0][0][ch][ro][co];
                            // std::cout << input0[ti][ba][ch][ro][co] << " + " << input1[0][0][ch][ro][co] << " = " << output[ti][ba][ch][ro][co] << '\n';
                        }
                    }
                }
            }
        }
        // }

        // GetOutput()->PrintData();

        return true;
    }

    virtual bool ComputeBackPropagate() {
        // std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int *shape = this->GetOutput()->GetShape();
        // int *shape_Input1   = GetInputOperator()[1]->GetOutput()->GetShape();
        DTYPE *****delta   = this->GetDelta()->GetData();
        this->GetInputOperator()[0]->GetDelta()->Reset();
        DTYPE *****_delta0 = this->GetInputOperator()[0]->GetDelta()->GetData();
        this->GetInputOperator()[1]->GetDelta()->Reset();
        DTYPE *****_delta1 = this->GetInputOperator()[1]->GetDelta()->GetData();


        // Tensor<DTYPE> *x1 = Tensor::Constants(shape_Input0[0], shape_Input0[1], shape_Input0[2], shape_Input0[3], shape_Input0[4], 1);
        //
        // delta = x1->GetData();

        // if ((shape_Input0[0] == shape_Input1[0]) && (shape_Input0[1] == shape_Input1[1])) {
        // for (int ti = 0; ti < shape_Input0[0]; ti++) {
        // for (int ba = 0; ba < shape_Input0[1]; ba++) {
        // for (int ch = 0; ch < shape_Input0[2]; ch++) {
        // for (int ro = 0; ro < shape_Input0[3]; ro++) {
        // for (int co = 0; co < shape_Input0[4]; co++) {
        // _delta0[ti][ba][ch][ro][co] = _delta1[ti][ba][ch][ro][co] = delta[ti][ba][ch][ro][co];
        // }
        // }
        // }
        // }
        // }
        // } else if ((shape_Input0[1] != shape_Input1[1]) && (shape_Input1[1] == 1)) {
        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            _delta0[ti][ba][ch][ro][co] = delta[ti][ba][ch][ro][co];
                            _delta1[0][0][ch][ro][co]  += delta[ti][ba][ch][ro][co];
                        }
                    }
                }
            }
        }
        // }

        // GetInputOperator()[0]->GetDelta()->PrintData();

        // GetInputOperator()[1]->GetDelta()->PrintData();

        // GetDelta()->Reset();
        return true;
    }
};

#endif  // ADD_H_
