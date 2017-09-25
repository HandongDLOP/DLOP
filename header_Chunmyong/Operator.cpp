#include "Operator.h"


// 만약 NeuralNetwork가 가진 Operator의 주소가 Output Layer일 경우

bool Operator::ForwardPropagate() {
    // Postorder
    for (int i = 0; i < m_OutputDgree; i++) {
        if (m_OutputOperator != NULL) m_OutputOperator->ForwardPropagate();
    }

    this->ExcuteForwardPropagate();

    return true;
}

bool Operator::BackPropagate() {
    // Preorder
    this->ExcuteBackPropagate();

    for (int i = 0; i < m_InputDegree; i++) {
        if (m_OutputOperator != NULL) m_OutputOperator->BackPropagate();
    }
    return true;
}

bool Operator::ExcuteForwardPropagate() {
    return true;
}

bool Operator::ExcuteBackPropagate() {
    return true;
}
