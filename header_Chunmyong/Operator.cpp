#include <iostream>
#include "Operator.h"

// 부모 클래스
bool Operator::Alloc(Ark *pInput, MetaParameter *pParam) {


    return true;
}

void Operator::Delete() {
    std::cout << "Operator::Delete()" << '\n';
}

/* BFS로 다시 구현할 필요 있음 */
bool Operator::ForwardPropagate() {
    // Postorder
    for (int i = 0; i < m_OutputDgree; i++) {
        if (m_pOutputOperator[i] != NULL) m_pOutputOperator[i]->ForwardPropagate();
    }

    this->ComputeForwardPropagate();

    return true;
}

bool Operator::BackPropagate() {
    // Preorder
    this->ComputeBackPropagate();

    for (int i = 0; i < m_InputDegree; i++) {
        if (m_pOutputOperator[i] != NULL) m_pOutputOperator[i]->BackPropagate();
    }
    return true;
}

bool Operator::ComputeForwardPropagate() {
    std::cout << "ComputeForwardPropagate" << '\n';
    return true;
}

bool Operator::ComputeBackPropagate() {
    std::cout << "ComputeBackPropagate" << '\n';
    return true;
}
