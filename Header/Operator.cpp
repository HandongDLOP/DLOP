#include "Operator.h"

// 부모 클래스
bool Operator::Alloc(Tensor *pTensor) {
    std::cout << "Operator::Alloc(Tensor *)" << '\n';

    m_pInputDim = new TensorShape *[1];
    m_aInput    = new Tensor *[1]; // 이것도 자동화 시킬 필요 있음
    return true;
}

bool Operator::Alloc(TensorShape *pshape) {
    std::cout << "Operator::Alloc(Tensor *)" << '\n';

    m_pInputDim = new TensorShape *[1];

    // for base operator
    m_aInput = new Tensor *[1];

    return true;
}

bool Operator::Alloc(Operator *pInput) {
    std::cout << "Operator::Alloc(Operator *)" << '\n';

    m_pInputDim = new TensorShape *[1];

    // 추후 자동 계산이 가능하게 구현 예정
    m_aInput = new Tensor *[1];

    // Shape도 받을 수 있도록 코드 수정 alloc도 마찬가지
    AddEdgebetweenOperators(pInput);

    return true;
}

bool Operator::Alloc(Operator *pInput1, Operator *pInput2) {
    std::cout << "Operator::Alloc(Operator *, Operator *)" << '\n';

    m_pInputDim = new TensorShape *[2];

    // 추후 자동 계산이 가능하게 구현 예정
    m_aInput = new Tensor *[2];

    // Shape도 받을 수 있도록 코드 수정 alloc도 마찬가지
    AddEdgebetweenOperators(pInput1);
    AddEdgebetweenOperators(pInput2);

    return true;
}

bool Operator::Alloc(MetaParameter *pParam) {
    return true;
}

bool Operator::AllocOptimizer(Optimizer_name pOptimizer_name) {
    for (int i = 0; i < m_InputDegree; i++) {
        m_aInputOperator[i]->AllocOptimizer(pOptimizer_name);

        Optimizer *pOptimizer = Factory::OptimizerFactory(pOptimizer_name);
        m_aInputOperator[i]->SetOptimizer(pOptimizer);
    }

    return true;
}

void Operator::Delete() {
    std::cout << "Operator::Delete()" << '\n';

    delete[] m_aInput;
    delete m_aOutput;
    delete m_aGradient;
    delete m_aDelta;
    delete[] m_aOutputOperator;
    delete[] m_aInputOperator;
    delete m_aOptimizer;
}

bool Operator::DeleteInputOperator() {
    // Postorder : like ForwardPropagate
    for (int i = 0; i < m_InputDegree; i++) {
        m_aInputOperator[i]->IncreaseCurrentOutputDegree();

        if (m_aInputOperator[i]->GetCurrentOutputDegree() == 1) {
            m_aInputOperator[i]->DeleteInputOperator();
        }

        if (m_aInputOperator[i]->GetOutputDegree() == m_aInputOperator[i]->GetCurrentOutputDegree()) {
            // std::cout << '\n' << m_aInputOperator[i]->GetName() << '\n' << std::endl;
            delete m_aInputOperator[i];
        }
    }

    return true;
}

// ===========================================================================================

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

bool Operator::AddEdgebetweenOperators(Operator *pInput) {
    // 양방향 Edge 생성
    _AddInputEdge(pInput);
    pInput->_AddOutputEdge(this);

    SetInput(pInput->GetOutput(), m_InputDegree - 1);

    return true;
}

// ===========================================================================================

/* BFS로 다시 구현할 필요 있음 */

// bool Operator::ForwardPropagate(){
//     // Preorder
//     this->ComputeForwardPropagate();
//
//     // value 조정
//     for (int i = 0; i < m_OutputDegree; i++) {
//         if (m_aOutputOperator[i] != NULL) m_aOutputOperator[i]->IncreaseCurrentInputDegree();
//     }
//     m_currentInputDegree = 0;
//
//     for (int i = 0; i < m_OutputDegree; i++) {
//         if (m_aOutputOperator[i]->GetInputDegree() == m_aOutputOperator[i]->GetCurrentInputDegree()) {
//             m_aOutputOperator[i]->ForwardPropagate();
//         }
//     }
//     return true;
// }

bool Operator::ForwardPropagate() {
    // Postorder

    // BFS로 얼른 바꾸자.... 현재 좋은 생각이 떠오르지 않음...
    // ForwardPropagate가 중복으로 실행되는 것만 막은 상황
    if ((m_currentOutputDegree == 0) || (m_currentOutputDegree == 1)) {
        for (int i = 0; i < m_InputDegree; i++) {
            m_aInputOperator[i]->IncreaseCurrentOutputDegree();
            m_aInputOperator[i]->ForwardPropagate();
        }

        this->ComputeForwardPropagate();
    }

    if (m_OutputDegree == m_currentOutputDegree) {
        m_currentOutputDegree = 0;
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

// ===========================================================================================

bool Operator::ComputeForwardPropagate() {
    std::cout << m_name << " : ComputeForwardPropagate()" << '\n';

    return true;
}

bool Operator::ComputeBackPropagate() {
    std::cout << m_name << " : ComputeBackPropagate()" << '\n';

    return true;
}

// ===========================================================================================

Operator * Operator::CheckEndOperator() {
    if (m_OutputDegree == 0) {
        return this;
    } else {
        // 추후에는 모든 Operator의 Output을 확인하도록 한다.
        return m_aOutputOperator[0]->CheckEndOperator();
    }
}
