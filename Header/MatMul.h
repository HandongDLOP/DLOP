#ifndef MATMUL_H_
#define MATMUL_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class MatMul : public Operator {
public:

    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (MatMul::Alloc())
    MatMul(Operator *pInput1, Operator *pInput2) : Operator(pInput1, pInput2) {
        std::cout << "MatMul::MatMul(Operator *, MetaParameter *)" << '\n';
        Alloc(pInput1, pInput2);
    }

    MatMul(Operator *pInput1, Operator *pInput2, std::string pName) : Operator(pInput1, pInput2, pName) {
        std::cout << "MatMul::MatMul(Operator *, MetaParameter *, std::string)" << '\n';
        Alloc(pInput1, pInput2);
    }

    virtual ~MatMul() {
        std::cout << "MatMul::~MatMul()" << '\n';
    }

    virtual bool Alloc(Operator *pInput1, Operator *pInput2) {
        std::cout << "MatMul::Alloc(Operator *, Operator *)" << '\n';

        TensorShape *InputDim0 = pInput1->GetOutput()->Getshape();
        TensorShape *InputDim1 = pInput2->GetOutput()->Getshape();

        // if pInput1 and pInput2의 shape가 다르면 abort
        if (InputDim0->Getdim()[1] != InputDim1->Getdim()[0]) {
            std::cout << InputDim0->Getdim()[1] << ", " << InputDim1->Getdim()[0] << '\n';
            std::cout << "data has invalid dimension" << '\n';
            exit(0);
        }

        int output_row = InputDim0->Getdim()[0];
        int output_col = InputDim1->Getdim()[1];

        // 결과물 shape (m by n @ n by k => m by k)
        TensorShape temp_shape(output_row, output_col, 0, 0, 0);
        SetOutput(new Tensor(&temp_shape));
        // Gradient는 Trainable한 요소에서만 필요하다.
        SetDelta(new Tensor(&temp_shape));

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        TensorShape *InputDim0 = GetInputOperator()[0]->GetOutput()->Getshape();
        TensorShape *InputDim1 = GetInputOperator()[1]->GetOutput()->Getshape();

        int row    = InputDim0->Getdim()[0];
        int hidden = InputDim0->Getdim()[1];
        int col    = InputDim1->Getdim()[1];

        float *input_data = GetInputOperator()[0]->GetOutput()->GetData();
        float *Weight     = GetInputOperator()[1]->GetOutput()->GetData();
        float *output_data     = GetOutput()->GetData();

        // float *output_data     = new float[row * col];
        float temp = 0.0;

        for (int cur_row = 0; cur_row < row; cur_row++) {
            for (int cur_col = 0; cur_col < col; cur_col++) {
                for (int hid = 0; hid < hidden; hid++) {
                    temp += input_data[hidden * cur_row + hid] * Weight[col * hid + cur_col];
                }
                output_data[col * cur_row + cur_col] = temp;
                temp                      = 0;
            }
        }

        // SetOutput(output_data);

        return true;
    }

    virtual bool ComputeBackPropagate() {
        std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        int output_col = GetOutput()->Getshape()->Getdim()[1];

        int size_input  = GetInputOperator()[0]->GetOutput()->GetFlatDim();
        int size_Weight = GetInputOperator()[1]->GetOutput()->GetFlatDim();

        float *input_data = GetInputOperator()[0]->GetOutput()->GetData(); // input_data
        float *Weight     = GetInputOperator()[1]->GetOutput()->GetData(); // Weight

        GetInputOperator()[0]->GetOutput()->PrintData();
        GetInputOperator()[1]->GetOutput()->PrintData();

        float *delta = GetDelta()->GetData();

        GetDelta()->PrintData();

        float *_delta_input  = new float[size_input];  // for weight
        float *_delta_Weight = new float[size_Weight]; // for input

        // 초기화 (나중에 코드 전체에 초기화 코드를 둘 것)
        for (int i = 0; i < size_input; i++) {
            _delta_input[i] = 0;
        }

        for (int i = 0; i < size_Weight; i++) {
            _delta_Weight[i] = 0;
        }

        for (int i = 0; i < size_Weight; i++) {
            _delta_input[i / output_col] += delta[i % output_col] * Weight[i];
            _delta_Weight[i]              = delta[i % output_col] * input_data[i / output_col];

            // _delta_input[i]           = delta[i / hidden] * Weight[i % hidden];
            // _delta_Weight[i % hidden] += delta[i / hidden] * input_data[i];
        }

        GetInputOperator()[0]->SetDelta(_delta_input);

        GetInputOperator()[0]->GetDelta()->PrintData();

        GetInputOperator()[1]->SetDelta(_delta_Weight);

        GetInputOperator()[1]->GetDelta()->PrintData();

        return true;
    }
};

#endif // MATMUL_H_
