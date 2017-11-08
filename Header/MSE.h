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

        if (m_pInputDim0->Getdim()[0] != m_pInputDim1->Getdim()[0]) {
            std::cout << "data has invalid dimension" << '\n';
            exit(0);
        }

        GetInput()[0]->PrintData();

        return true;
    }

    virtual bool ComputeBackPropagate() {
        std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        TensorShape *m_pInputDim0 = GetInputDim()[0];
        TensorShape *m_pInputDim1 = GetInputDim()[1];

        if (m_pInputDim0->Getdim()[0] != m_pInputDim1->Getdim()[0]) {
            std::cout << "data has invalid dimension" << '\n';
            exit(0);
        }

        int output    = m_pInputDim0->Getdim()[0];
        int batch    = m_pInputDim0->Getdim()[1];

        float *data0  = GetInput()[0]->GetData();
        float *data1  = GetInput()[1]->GetData();
        float *result = new float[output * batch];

        for(int i = 0; i < output; i++){
            result[i] = (data0[i] - data1[i]) / output;
        }

        GetInputOperator()[0]->SetDelta(result);

        GetInputOperator()[0]->GetDelta()->PrintData();

        return true;
    }
};

#endif  // MSE_H_
