#include "Operator.h"

// 부모 클래스
bool Operator::Alloc(Tensor *pTensor) {
    std::cout << "Operator::Alloc(Tensor *)" << '\n';

    return true;
}

bool Operator::Alloc(Operator *pInput) {
    std::cout << "Operator::Alloc(Operator *)" << '\n';

    // Shape도 받을 수 있도록 코드 수정 alloc도 마찬가지
    AddEdgebetweenOperators(pInput);

    return true;
}

bool Operator::Alloc(Operator *pInput0, Operator *pInput1) {
    std::cout << "Operator::Alloc(Operator *, Operator *)" << '\n';

    // Shape도 받을 수 있도록 코드 수정 alloc도 마찬가지
    AddEdgebetweenOperators(pInput0);
    AddEdgebetweenOperators(pInput1);

    return true;
}

bool Operator::Alloc(MetaParameter *pParam) {
    return true;
}

// bool Operator::AllocOptimizer(Optimizer_name pOptimizer_name) {
// for (int i = 0; i < m_InputDegree; i++) {
// m_apInputOperator[i]->AllocOptimizer(pOptimizer_name);
//
// Optimizer *pOptimizer = Factory::OptimizerFactory(pOptimizer_name);
// m_apInputOperator[i]->SetOptimizer(pOptimizer);
// }
//
// return true;
// }

bool Operator::AllocOptimizer(Optimizer *pOptimizer) {
    for (int i = 0; i < m_InputDegree; i++) {
        m_apInputOperator[i]->AllocOptimizer(pOptimizer);

        if (m_apInputOperator[i]->GetTrainable() == 1) {
            pOptimizer->AddTrainableData(m_apInputOperator[i]->GetOutput(), m_apInputOperator[i]->GetGradient());
        }
    }

    return true;
}

void Operator::Delete() {
    std::cout << "Operator::Delete()" << '\n';

    delete m_aOutput;
    delete m_aGradient;
    delete m_aDelta;
    delete[] m_apOutputOperator;
    delete[] m_apInputOperator;
    // delete m_aOptimizer;
}

bool Operator::DeleteInputOperator() {
    // Postorder : like ForwardPropagate
    for (int i = 0; i < m_InputDegree; i++) {
        m_apInputOperator[i]->IncreaseCurrentOutputDegree();

        if (m_apInputOperator[i]->GetCurrentOutputDegree() == 1) {
            m_apInputOperator[i]->DeleteInputOperator();
        }

        if (m_apInputOperator[i]->GetOutputDegree() == m_apInputOperator[i]->GetCurrentOutputDegree()) {
            // std::cout << '\n' << m_apInputOperator[i]->GetName() << '\n' << std::endl;
            delete m_apInputOperator[i];
        }
    }

    return true;
}

// ===========================================================================================

// Add Graph Edge
bool Operator::_AddInputEdge(Operator *pInput) {
    if (m_InputDegree != 0) {
        Operator **temp = new Operator *[m_InputDegree + 1];
        std::copy(m_apInputOperator, m_apInputOperator + m_InputDegree, temp);

        delete[] m_apInputOperator;

        m_apInputOperator = temp;
    } else {
        m_apInputOperator = new Operator *[m_InputDegree + 1];
    }

    m_apInputOperator[m_InputDegree] = pInput;

    m_InputDegree++;

    return true;
}

bool Operator::_AddOutputEdge(Operator *pOutput) {
    if (m_OutputDegree != 0) {
        Operator **temp = new Operator *[m_OutputDegree + 1];
        std::copy(m_apOutputOperator, m_apOutputOperator + m_OutputDegree, temp);

        delete[] m_apOutputOperator;

        m_apOutputOperator = temp;
    } else {
        m_apOutputOperator = new Operator *[m_OutputDegree + 1];
    }

    m_apOutputOperator[m_OutputDegree] = pOutput;

    m_OutputDegree++;

    return true;
}

bool Operator::AddEdgebetweenOperators(Operator *pInput) {
    // 양방향 Edge 생성
    _AddInputEdge(pInput);
    pInput->_AddOutputEdge(this);

    return true;
}

// ===========================================================================================

/* BFS로 다시 구현할 필요 있음 */

// bool Operator::ForwardPropagate(){
//// Preorder
// this->ComputeForwardPropagate();
//
//// value 조정
// for (int i = 0; i < m_OutputDegree; i++) {
// if (m_apOutputOperator[i] != NULL) m_apOutputOperator[i]->IncreaseCurrentInputDegree();
// }
// m_currentInputDegree = 0;
//
// for (int i = 0; i < m_OutputDegree; i++) {
// if (m_apOutputOperator[i]->GetInputDegree() == m_apOutputOperator[i]->GetCurrentInputDegree()) {
// m_apOutputOperator[i]->ForwardPropagate();
// }
// }
// return true;
// }

bool Operator::ForwardPropagate() {
    // 알고리즘 잘 이해하기
    // BFS로 나중에는 바꿀 것
    if (m_InputDegree == m_currentInputDegree) {
        this->ComputeForwardPropagate();

        for (int o = 0; o < m_OutputDegree; o++) {
            m_apOutputOperator[o]->IncreaseCurrentInputDegree();
        }
        m_currentInputDegree = 0;
    } else {
        for (int i = 0; i < m_InputDegree; i++) {
            m_apInputOperator[i]->ForwardPropagate();

            if (m_InputDegree == m_currentInputDegree) {
                this->ComputeForwardPropagate();

                for (int o = 0; o < m_OutputDegree; o++) {
                    m_apOutputOperator[o]->IncreaseCurrentInputDegree();
                }
                m_currentInputDegree = 0;
            }
        }
    }

    return true;
}

bool Operator::BackPropagate() {
    this->ComputeBackPropagate();

    // value 조정
    for (int i = 0; i < m_InputDegree; i++) {
        if (m_apInputOperator[i] != NULL) m_apInputOperator[i]->IncreaseCurrentOutputDegree();
    }
    m_currentOutputDegree = 0;

    // Back propagation을 하다가 base operator가 나오지 않으면, 실행되지 않은 Placeholder가 있지는 않은지 확인해볼 것
    for (int i = 0; i < m_InputDegree; i++) {
        if (m_apInputOperator[i]->GetOutputDegree() == m_apInputOperator[i]->GetCurrentOutputDegree()) {
            m_apInputOperator[i]->BackPropagate();
        }
    }
    return true;
}

// ===========================================================================================

bool Operator::ComputeForwardPropagate() {
    // std::cout << m_name << " : ComputeForwardPropagate()" << '\n';

    return true;
}

bool Operator::ComputeBackPropagate() {
    // std::cout << m_name << " : ComputeBackPropagate()" << '\n';

    return true;
}

// ===========================================================================================

Operator * Operator::CheckEndOperator() {
    // recursively
    if (m_OutputDegree == 0) {
        return this;
    } else {
        // 추후에는 모든 Operator의 Output을 확인하도록 한다.
        return m_apOutputOperator[0]->CheckEndOperator();
    }
}
