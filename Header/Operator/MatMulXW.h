#ifndef MATMUL_H_
#define MATMUL_H_    value

#include "..//Operator.h"

class MatMul : public Operator {
public:
    MatMul(Operator *pInput0, Operator *pInput1) : Operator(pInput0, pInput1) {
        std::cout << "MatMul::MatMul(Operator *, MetaParameter *)" << '\n';
        Alloc(pInput0, pInput1);
    }

    MatMul(Operator *pInput0, Operator *pInput1, std::string pName) : Operator(pInput0, pInput1, pName) {
        std::cout << "MatMul::MatMul(Operator *, MetaParameter *, std::string)" << '\n';
        Alloc(pInput0, pInput1);
    }

    virtual ~MatMul() {
        std::cout << "MatMul::~MatMul()" << '\n';
    }

    virtual bool Alloc(Operator *pInput0, Operator *pInput1) {
        std::cout << "MatMul::Alloc(Operator *, Operator *)" << '\n';

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

        // factory method 작업

        // if (shape_Input0[1] != shape_Input1[1]) {
        // if ((shape_Input0[1] == 1) || (shape_Input1[1] == 1)) {
        // if (shape_Input0[1] < shape_Input1[1]) {
        // GetInputOperator()[0] = pInput1;
        // GetInputOperator()[1] = pInput0;
        // }
        // } else {
        // std::cout << "data has unvalid batch dimension" << '\n';
        // exit(0);
        // }
        // }

        // if(shape_Input0[1] > shape_Input1[1] && shape_Input0[0] > shape_Input1[0]){
        // int time = pInput0->GetOutput()->GetBatch();
        // int batch = pInput0->GetOutput()->GetBatch();
        // } else if(shape_Input0[1] < shape_Input1[1] && shape_Input0[0] < shape_Input1[0]){
        // int time = pInput1->GetOutput()->GetBatch();
        // int batch = pInput1->GetOutput()->GetBatch();
        // } else {
        // std::cout << "invalid dimension" << '\n';
        // exit(0);
        // }

        // w * x 의 형태에서만 진행
        int Time    = pInput0->GetOutput()->GetTime();
        int Batch   = pInput0->GetOutput()->GetBatch();
        int Channel = pInput0->GetOutput()->GetChannel();
        int Row     = pInput0->GetOutput()->GetRow();
        int Col     = pInput1->GetOutput()->GetCol();

        // 결과물 shape (m by n @ n by k => m by k)
        SetOutput(new Tensor(Time, Batch, Channel, Row, Col));
        // Gradient는 Trainable한 요소에서만 필요하다.
        SetDelta(new Tensor(Time, Batch, Channel, Row, Col));

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        // std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        int Time    = GetOutput()->GetTime();
        int Batch   = GetOutput()->GetBatch();
        int Channel = GetOutput()->GetChannel();
        int Row     = GetOutput()->GetRow();
        int Col     = GetOutput()->GetCol();
        int Hidden  = GetInputOperator()[0]->GetOutput()->GetCol();

        double *****input0 = GetInputOperator()[0]->GetOutput()->GetData();  // weight
        double *****input1 = GetInputOperator()[1]->GetOutput()->GetData();  // input
        double *****output = GetOutput()->GetData();

        // double *****output_data     = new float[row * col];
        double temp = 0.0;

        // GetInputOperator()[0]->GetOutput()->PrintShape();
        // GetInputOperator()[1]->GetOutput()->PrintShape();
        // GetOutput()->PrintShape();

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

        // SetOutput(output_data);

        return true;
    }

    virtual bool ComputeBackPropagate() {
        // std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int Time    = GetOutput()->GetTime();
        int Batch   = GetOutput()->GetBatch();
        int Channel = GetOutput()->GetChannel();
        int Row     = GetOutput()->GetRow();
        int Col     = GetOutput()->GetCol();
        int Hidden  = GetInputOperator()[0]->GetOutput()->GetCol();

        double *****input0 = GetInputOperator()[0]->GetOutput()->GetData();  // weight
        double *****input1 = GetInputOperator()[1]->GetOutput()->GetData();  // input

        // GetInputOperator()[0]->GetOutput()->PrintData();
        // GetInputOperator()[1]->GetOutput()->PrintData();

        double *****delta = GetDelta()->GetData();

        // GetDelta()->PrintData();

        GetInputOperator()[0]->GetDelta()->Reset();
        GetInputOperator()[1]->GetDelta()->Reset();
        double *****delta_input0 = GetInputOperator()[0]->GetDelta()->GetData();  // weight
        double *****delta_input1 = GetInputOperator()[1]->GetDelta()->GetData();  // input

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
        // for (int i = 0; i < size_Weight; i++) {
        // _delta_input[i / output_col] += delta[i % output_col] * Weight[i];
        // _delta_Weight[i]              = delta[i % output_col] * input_data[i / output_col];
        // }

        // GetInputOperator()[0]->GetDelta()->PrintData();

        // GetInputOperator()[1]->GetDelta()->PrintData();

        // GetDelta()->Reset();

        return true;
    }
};

#endif  // MATMUL_H_
