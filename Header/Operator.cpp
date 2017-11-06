#include <iostream>
#include <algorithm>

#include "Operator.h"

// 부모 클래스
bool Operator::Alloc(Tensor *pTensor) {
    std::cout << "Operator::Alloc(Tensor *)" << '\n';

    m_aOutput = pTensor;

    return true;
}

bool Operator::Alloc(Operator *pInput) {
    std::cout << "Operator::Alloc(Operator *)" << '\n';

    // 쌍방향 연결관계 추가
    _AddInputEdge(pInput);
    pInput->_AddOutputEdge(this);

    m_pInput  = pInput->GetOutput();
    m_aOutput = new Tensor();

    return true;
}

bool Operator::Alloc(MetaParameter *pParam) {
    return true;
}

void Operator::Delete() {
    std::cout << "Operator::Delete()" << '\n';

    delete m_aOutput;
    delete m_aWeight;
    delete m_aGradient;
    delete m_aDelta;
    delete [] m_aOutputOperator;
    delete [] m_aInputOperator;
}

bool Operator::PropagateDelete() {
    // Postorder : like ForwardPropagate
    for (int i = 0; i < m_InputDegree; i++) {
        m_aInputOperator[i]->PropagateDelete();
        delete m_aInputOperator[i];
    }
    return true;
}

//===========================================================================================

// Add Graph Edge
bool Operator::_AddInputEdge(Operator *pInput) {
    if (m_InputDegree != 0) {
        Operator **temp = new Operator *[m_InputDegree + 1];
        std::copy(m_aInputOperator, m_aInputOperator + m_InputDegree, temp);

        delete[] m_aInputOperator;

        m_aInputOperator = temp;
    } else {
        m_aInputOperator = new Operator *[m_InputDegree + 1];
    }

    m_aInputOperator[m_InputDegree] = pInput;

    m_InputDegree++;

    return true;
}

bool Operator::_AddOutputEdge(Operator *pOutput) {
    if (m_OutputDegree != 0) {
        Operator **temp = new Operator *[m_OutputDegree + 1];
        std::copy(m_aOutputOperator, m_aOutputOperator + m_OutputDegree, temp);

        delete[] m_aOutputOperator;

        m_aOutputOperator = temp;
    } else {
        m_aOutputOperator = new Operator *[m_OutputDegree + 1];
    }

    m_aOutputOperator[m_OutputDegree] = pOutput;

    m_OutputDegree++;

    return true;
}

//===========================================================================================

/* BFS로 다시 구현할 필요 있음 */

bool Operator::ForwardPropagate() {
    // Postorder
    for (int i = 0; i < m_InputDegree; i++) {
        m_aInputOperator[i]->ForwardPropagate();
    }

    if (this->GetInputDegree() == this->GetCurrentInputDegree()) {
        this->ComputeForwardPropagate();
        // value 조정
        for (int i = 0; i < m_OutputDegree; i++) {
            if (m_aOutputOperator[i] != NULL) m_aOutputOperator[i]->IncreaseCurrentInputDegree();
        }
        m_currentInputDegree = 0;
    }

    return true;
}

bool Operator::BackPropagate() {
    // Preorder
    this->ComputeBackPropagate();
    // value 조정
    for (int i = 0; i < m_InputDegree; i++) {
        if (m_aInputOperator[i] != NULL) m_aInputOperator[i]->IncreaseCurrentOutputDegree();
    }
    m_currentOutputDegree = 0;

    // Back propagation을 하다가 base operator가 나오지 않으면, 실행되지 않은 Placeholder가 있지는 않은지 확인해볼 것
    for (int i = 0; i < m_InputDegree; i++) {
        if (m_aInputOperator[i]->GetOutputDegree() == m_aInputOperator[i]->GetCurrentOutputDegree()) {
            m_aInputOperator[i]->BackPropagate();
        }
    }
    return true;
}

//===========================================================================================

bool Operator::ComputeForwardPropagate() {
    std::cout << m_name << " : ComputeForwardPropagate()" << '\n';

    return true;
}

bool Operator::ComputeBackPropagate() {
    std::cout << m_name << " : ComputeBackPropagate()" << '\n';

    return true;
}
