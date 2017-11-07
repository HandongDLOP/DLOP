#ifndef COPY_H_
#define COPY_H_    value

#include <iostream>
#include <string>

#include "Tensor.h"
#include "Operator.h"

class Copy : public Operator {
public:
    Copy(Operator *pInput) : Operator(pInput) {
        std::cout << "Copy::Copy(Operator *, MetaParameter *)" << '\n';
        // Alloc(pInput, pParam);
    }

    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Copy::Alloc())
    Copy(Operator *pInput, std::string pName) : Operator(pInput, pName) {
        std::cout << "Copy::Copy(Operator *, MetaParameter *)" << '\n';
        Alloc(pInput);
    }

    virtual ~Copy() {
        std::cout << "Copy::~Copy()" << '\n';
    }

    virtual bool Alloc(Operator *pInput) {
        std::cout << "Copy::Alloc(Operator *)" << '\n';

        // Output의 dimension은 매우 중요하며, 그래프를 만들 때는 무조건 dimension을 미리 Alloc할 때 정할 필요가 있다.

        Tensor * temp_output = new Tensor(GetInput()[0]->Getshape());

        SetOutput(temp_output);  // Input dimesion 과 동일

        Tensor * temp_delta = new Tensor(GetInput()[0]->Getshape());

        SetDelta(temp_delta);  // 순서는 Delta에 먼저 저장되고, 그 후에 Gradient에 더해진다.

        // SetDelta(GetOutput()->Getshape()); // 이 경우는 아직 Output dimension을 몰라서 오류를 말생시킨다.

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';

        SetOutput(GetInput()[0]->GetData());

        GetOutput()->PrintData();

        return true;
    }
};

#endif  // COPY_H_
