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
        // if pInput1 and pInput2의 shape가 다르면 abort

        // Input dim 저장
        SetInputDim(pInput1->GetOutput()->Getshape(), 0);
        SetInputDim(pInput2->GetOutput()->Getshape(), 1);

        TensorShape *m_pInputDim0 = GetInputDim()[0];
        TensorShape *m_pInputDim1 = GetInputDim()[1];

        // 결과물 shape (m by n @ n by k => m by k)
        TensorShape *temp_shape = new TensorShape(m_pInputDim0->Getdim()[0], m_pInputDim1->Getdim()[1], 0, 0, 0);

        Tensor *temp_output = new Tensor(temp_shape);

        SetOutput(temp_output);

        SetOutputDim(GetOutput()->Getshape());

        // Gradient는 Trainable한 요소에서만 필요하다.

        // delta는 무조건 Output dim을 따르며, 무조건 위 Operator에서 계산되어 내려오게 된다.
        Tensor *temp_delta = new Tensor(temp_shape);

        SetDelta(temp_delta);

        delete temp_shape;

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        TensorShape *m_pInputDim0 = GetInputDim()[0];
        TensorShape *m_pInputDim1 = GetInputDim()[1];

        if (m_pInputDim0->Getdim()[1] != m_pInputDim1->Getdim()[0]) {
            std::cout << "data has invalid dimension" << '\n';
            exit(0);
        }

        int row    = m_pInputDim0->Getdim()[0];
        int hidden = m_pInputDim0->Getdim()[1];
        int col    = m_pInputDim1->Getdim()[1];

        float *data0  = GetInput()[0]->GetData();
        float *data1  = GetInput()[1]->GetData();
        float *result = new float[row * col];
        float  temp   = 0.0;

        for (int row0 = 0; row0 < row; row0++) {
            for (int col1 = 0; col1 < col; col1++) {
                for (int hid = 0; hid < hidden; hid++) {
                    temp += data0[hidden * row0 + hid] * data1[col * hid + col1];
                }
                result[col * row0 + col1] = temp;
                temp                      = 0;
            }
        }

        SetOutput(result);

        return true;
    }

    virtual bool ComputeBackPropagate() {
        std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        // TensorShape *m_pInputDim0 = GetInputDim()[0];
        TensorShape *m_pInputDim1 = GetInputDim()[1];

        // int row    = m_pInputDim0->Getdim()[0];
        int hidden = m_pInputDim1->Getdim()[0];

        int size0 = GetInput()[0]->GetFlatDim();
        int size1 = GetInput()[1]->GetFlatDim();

        float *data0 = GetInput()[0]->GetData();  // Weight
        float *data1 = GetInput()[1]->GetData();  // input

        GetInput()[0]->PrintData();
        GetInput()[1]->PrintData();

        // // Test code
        // Tensor * temp = Tensor::Constants(5, 1, 0, 0, 0,1.0);
        //
        // temp->PrintData();
        //
        // float *delta = temp->GetData();

        float *delta = GetDelta()->GetData();

        float *_delta0 = new float[size0];  // for weight
        float *_delta1 = new float[size1];  // for input

        //초기화 (나중에 코드 전체에 초기화 코드를 둘 것)
        for (int i = 0; i < size1; i++){
            _delta1[i] = 0;
        }

        for (int i = 0; i < size0; i++) {
            _delta0[i]           = delta[i / hidden] * data1[i % hidden];
            _delta1[i % hidden] += delta[i / hidden] * data0[i];
        }

        GetInputOperator()[0]->SetDelta(_delta0);

        GetInputOperator()[0]->GetDelta()->PrintData();

        GetInputOperator()[1]->SetDelta(_delta1);

        GetInputOperator()[1]->GetDelta()->PrintData();

        return true;
    }
};

#endif  // MATMUL_H_
