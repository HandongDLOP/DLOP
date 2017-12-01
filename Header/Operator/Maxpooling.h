#ifndef MAXPOOLING_H_
#define MAXPOOLING_H_    value

#include "..//Operator.h"

class Maxpooling : public Operator {
public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Maxpooling::Alloc())

    Maxpooling(Operator *pInput, MetaParameter *pParam, std::string pName) : Operator(pInput, pParam, pName) {
        std::cout << "Maxpooling::Maxpooling(Operator *, MetaParameter *, std::string)" << '\n';
        // Alloc(pInput, pParam);
    }

    virtual ~Maxpooling() {
        std::cout << "Maxpooling::~Maxpooling()" << '\n';
    }

    virtual bool Alloc(Operator *pInput, MetaParameter *pParam) {

        return true;
    }

    virtual bool ComputeForwardPropagate() {
        std::cout << GetName() << " : ComputeForwardPropagate()" << '\n';


        return true;
    }

    virtual bool ComputeBackPropagate() {
        std::cout << GetName() << " : ComputeBackPropagate()" << '\n';

        return true;
    }
};

#endif  // MAXPOOLING_H_