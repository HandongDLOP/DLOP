#include "Operator.h"

template class Operator<int>;
template class Operator<float>;
template class Operator<double>;

template<typename DTYPE>
bool Operator<DTYPE>::Alloc(Tensor<DTYPE> *pTensor) {
    std::cout << "Operator<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';

    return true;
}

template<typename DTYPE>
bool Operator<DTYPE>::Alloc(Operator<DTYPE> *pInput) {
    std::cout << "Operator<DTYPE>::Alloc(Operator<DTYPE> *)" << '\n';

    // Shape도 받을 수 있도록 코드 수정 alloc도 마찬가지
    this->AddEdgebetweenOperators(pInput);

    return true;
}

template<typename DTYPE>
bool Operator<DTYPE>::Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1) {
    std::cout << "Operator<DTYPE>::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

    // Shape도 받을 수 있도록 코드 수정 alloc도 마찬가지
    this->AddEdgebetweenOperators(pInput0);
    this->AddEdgebetweenOperators(pInput1);

    return true;
}

template<typename DTYPE>
bool Operator<DTYPE>::Alloc(MetaParameter<DTYPE> *pParam) {
    return true;
}

// bool Operator<DTYPE>::AllocOptimizer(Optimizer_name pOptimizer_name) {
// for (int i = 0; i < m_InputDegree; i++) {
// m_apInputOperator[i]->AllocOptimizer(pOptimizer_name);
//
// Optimizer<DTYPE> *pOptimizer = Factory::OptimizerFactory(pOptimizer_name);
// m_apInputOperator[i]->SetOptimizer(pOptimizer);
// }
//
// return true;
// }

template<typename DTYPE>
bool Operator<DTYPE>::AllocOptimizer(Optimizer<DTYPE> *pOptimizer) {
    for (int i = 0; i < m_InputDegree; i++) {
        m_apInputOperator[i]->AllocOptimizer(pOptimizer);

        if (m_apInputOperator[i]->GetTrainable() == 1) {
            pOptimizer->AddTrainableData(m_apInputOperator[i]->GetOutput(), m_apInputOperator[i]->GetGradient());
        }
    }

    return true;
}

template<typename DTYPE>
void Operator<DTYPE>::Delete() {
    std::cout << "Operator<DTYPE>::Delete()" << '\n';

    if (m_aOutput != NULL) delete m_aOutput;

    if (m_aGradient != NULL) delete m_aGradient;

    if (m_aDelta != NULL) delete m_aDelta;

    delete[] m_apOutputOperator;
    delete[] m_apInputOperator;
    // delete m_aOptimizer;
}

// bool Operator<DTYPE>::DeleteInputOperator() {
//// Postorder : like ForwardPropagate
// for (int i = 0; i < m_InputDegree; i++) {
// m_apInputOperator[i]->IncreaseCurrentOutputDegree();
//
// if (m_apInputOperator[i]->GetCurrentOutputDegree() == 1) {
// m_apInputOperator[i]->DeleteInputOperator();
// }
//
// if (m_apInputOperator[i]->GetOutputDegree() == m_apInputOperator[i]->GetCurrentOutputDegree()) {
//// std::cout << '\n' << m_apInputOperator[i]->GetName() << '\n' << std::endl;
// delete m_apInputOperator[i];
// }
// }
//
// return true;
// }

// ===========================================================================================

// Add Graph Edge
template<typename DTYPE>
bool Operator<DTYPE>::_AddInputEdge(Operator<DTYPE> *pInput) {
    if (m_InputDegree != 0) {
        Operator<DTYPE> **temp = new Operator<DTYPE> *[m_InputDegree + 1];
        std::copy(m_apInputOperator, m_apInputOperator + m_InputDegree, temp);

        delete[] m_apInputOperator;

        m_apInputOperator = temp;
    } else {
        m_apInputOperator = new Operator<DTYPE> *[m_InputDegree + 1];
    }

    m_apInputOperator[m_InputDegree] = pInput;

    m_InputDegree++;

    return true;
}

template<typename DTYPE>
bool Operator<DTYPE>::_AddOutputEdge(Operator<DTYPE> *pOutput) {
    if (m_OutputDegree != 0) {
        Operator<DTYPE> **temp = new Operator<DTYPE> *[m_OutputDegree + 1];
        std::copy(m_apOutputOperator, m_apOutputOperator + m_OutputDegree, temp);

        delete[] m_apOutputOperator;

        m_apOutputOperator = temp;
    } else {
        m_apOutputOperator = new Operator<DTYPE> *[m_OutputDegree + 1];
    }

    m_apOutputOperator[m_OutputDegree] = pOutput;

    m_OutputDegree++;

    return true;
}

template<typename DTYPE>
bool Operator<DTYPE>::AddEdgebetweenOperators(Operator<DTYPE> *pInput) {
    // 양방향 Edge 생성
    this->_AddInputEdge(pInput);
    pInput->_AddOutputEdge(this);

    return true;
}

// ===========================================================================================

/* BFS로 다시 구현할 필요 있음 */

// bool Operator<DTYPE>::ForwardPropagate(){
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

template<typename DTYPE>
bool Operator<DTYPE>::ForwardPropagate() {
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

template<typename DTYPE>
bool Operator<DTYPE>::BackPropagate() {
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

template<typename DTYPE>
bool Operator<DTYPE>::ComputeForwardPropagate() {
    // std::cout << m_name << " : ComputeForwardPropagate()" << '\n';

    return true;
}

template<typename DTYPE>
bool Operator<DTYPE>::ComputeBackPropagate() {
    // std::cout << m_name << " : ComputeBackPropagate()" << '\n';

    return true;
}

// ===========================================================================================

template<typename DTYPE>
void Operator<DTYPE>::PrintGraph(int depth) {
    std::cout << this->GetName() << '\n';

    // value 조정
    for (int i = 0; i < m_InputDegree; i++) {
        if (m_apInputOperator[i] != NULL) m_apInputOperator[i]->IncreaseCurrentOutputDegree();
    }
    m_currentOutputDegree = 0;

    // Back propagation을 하다가 base operator가 나오지 않으면, 실행되지 않은 Placeholder가 있지는 않은지 확인해볼 것
    for (int i = 0; i < m_InputDegree; i++) {
        if (m_apInputOperator[i]->GetOutputDegree() == m_apInputOperator[i]->GetCurrentOutputDegree()) {
            m_apInputOperator[i]->PrintGraph(depth);
        }
    }
}

template<typename DTYPE>
void Operator<DTYPE>::PrintData(int forceprint) {
    std::cout << "\n\n" << this->GetName() << ": PrintData()" << '\n';

    if (m_aOutput != NULL) this->PrintOutput(forceprint);

    if (m_aDelta != NULL) this->PrintDelta(forceprint);

    if (m_aGradient != NULL) this->PrintGradient(forceprint);

    std::cout << this->GetName() << ": ~PrintData() \n\n";

    // value 조정
    for (int i = 0; i < m_InputDegree; i++) {
        if (m_apInputOperator[i] != NULL) m_apInputOperator[i]->IncreaseCurrentOutputDegree();
    }
    m_currentOutputDegree = 0;

    // Back propagation을 하다가 base operator가 나오지 않으면, 실행되지 않은 Placeholder가 있지는 않은지 확인해볼 것
    for (int i = 0; i < m_InputDegree; i++) {
        if (m_apInputOperator[i]->GetOutputDegree() == m_apInputOperator[i]->GetCurrentOutputDegree()) {
            m_apInputOperator[i]->PrintData();
        }
    }
}

// ===========================================================================================

// template <typename DTYPE>
// Operator<DTYPE> * Operator<DTYPE>::CheckEndOperator() {
//// recursively
// if (m_OutputDegree == 0) {
// return this;
// } else {
//// 추후에는 모든 Operator의 Output을 확인하도록 한다.
// return m_apOutputOperator[0]->CheckEndOperator();
// }
// }
