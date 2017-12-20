#ifndef MATMUL_H_
#define MATMUL_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class MatMul : public Operator<DTYPE>{
private:
    typedef typename Tensor<DTYPE>::TENSOR_DTYPE TENSOR_DTYPE;

public:
    MatMul(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1) : Operator<DTYPE>(pInput0, pInput1) {
        std::cout << "MatMul::MatMul(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        this->Alloc(pInput0, pInput1);
    }

    MatMul(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        std::cout << "MatMul::MatMul(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        this->Alloc(pInput0, pInput1);
    }

    ~MatMul() {
        std::cout << "MatMul::~MatMul()" << '\n';
    }

    virtual int Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1) {
        std::cout << "MatMul::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        int *shape_Input0 = pInput0->GetOutput()->GetShape();
        int *shape_Input1 = pInput1->GetOutput()->GetShape();

        if (shape_Input0[4] != shape_Input1[3]) {
            std::cout << "data has different hidden dimension" << '\n';
            exit(0);
        }

        if (shape_Input0[2] != shape_Input1[2]) {
            std::cout << "data has different hidden dimension" << '\n';
            exit(0);
        }

        // w * x 의 형태에서만 진행
        int Time    = pInput0->GetOutput()->GetTime();
        int Batch   = pInput0->GetOutput()->GetBatch();
        int Channel = pInput0->GetOutput()->GetChannel();
        int Row     = pInput0->GetOutput()->GetRow();
        int Col     = pInput1->GetOutput()->GetCol();

        // 결과물 shape (m by n @ n by k => m by k)
        this->SetOutput(new Tensor<DTYPE>(Time, Batch, Channel, Row, Col));
        // Gradient는 Trainable한 요소에서만 필요하다.
        this->SetDelta(new Tensor<DTYPE>(Time, Batch, Channel, Row, Col));

        return 1;
    }

    virtual int ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int Time    = this->GetOutput()->GetTime();
        int Batch   = this->GetOutput()->GetBatch();
        int Channel = this->GetOutput()->GetChannel();
        int Row     = this->GetOutput()->GetRow();
        int Col     = this->GetOutput()->GetCol();
        int Hidden  = this->GetInputOperator()[0]->GetOutput()->GetCol();

        TENSOR_DTYPE input0 = this->GetInputOperator()[0]->GetOutput()->GetData();  // weight
        TENSOR_DTYPE input1 = this->GetInputOperator()[1]->GetOutput()->GetData();  // input
        TENSOR_DTYPE output = this->GetOutput()->GetData();

        DTYPE temp = 0.0;

        for (int ti = 0; ti < Time; ti++) {
            for (int ba = 0; ba < Batch; ba++) {
                for (int ch = 0; ch < Channel; ch++) {
                    for (int ro = 0; ro < Row; ro++) {
                        for (int co = 0; co < Col; co++) {
                            for (int hid = 0; hid < Hidden; hid++) {
                                temp += input0[ti][ba][ch][ro][hid] * input1[0][0][ch][hid][co];
                            }
                            output[ti][ba][ch][ro][co] = temp;
                            temp                       = 0.0;
                        }
                    }
                }
            }
        }

        return 1;
    }

    virtual int ComputeBackPropagate() {

        int Time    = this->GetOutput()->GetTime();
        int Batch   = this->GetOutput()->GetBatch();
        int Channel = this->GetOutput()->GetChannel();
        int Row     = this->GetOutput()->GetRow();
        int Col     = this->GetOutput()->GetCol();
        int Hidden  = this->GetInputOperator()[0]->GetOutput()->GetCol();

        TENSOR_DTYPE input0 = this->GetInputOperator()[0]->GetOutput()->GetData();  // input
        TENSOR_DTYPE input1 = this->GetInputOperator()[1]->GetOutput()->GetData();  // weight

        TENSOR_DTYPE delta = this->GetDelta()->GetData();

        this->GetInputOperator()[0]->GetDelta()->Reset();
        this->GetInputOperator()[1]->GetDelta()->Reset();
        TENSOR_DTYPE delta_input0 = this->GetInputOperator()[0]->GetDelta()->GetData();  // input
        TENSOR_DTYPE delta_input1 = this->GetInputOperator()[1]->GetDelta()->GetData();  // weight

        for (int ti = 0; ti < Time; ti++) {
            for (int ba = 0; ba < Batch; ba++) {
                for (int ch = 0; ch < Channel; ch++) {
                    for (int ro = 0; ro < Row; ro++) {
                        for (int co = 0; co < Col; co++) {
                            for (int hid = 0; hid < Hidden; hid++) {
                                delta_input0[ti][ba][ch][ro][hid] += input1[0][0][ch][hid][co] * delta[ti][ba][ch][ro][co];
                                delta_input1[0][0][ch][hid][co]   += input0[ti][ba][ch][ro][hid] * delta[ti][ba][ch][ro][co];
                            }
                        }
                    }
                }
            }
        }

        return 1;
    }
};

#endif  // MATMUL_H_
