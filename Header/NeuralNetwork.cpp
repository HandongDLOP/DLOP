#include "NeuralNetwork.h"

template class NeuralNetwork<int>;
template class NeuralNetwork<float>;
template class NeuralNetwork<double>;

template<typename DTYPE> NeuralNetwork<DTYPE>::NeuralNetwork() {
    std::cout << "NeuralNetwork<DTYPE>::NeuralNetwork()" << '\n';
    m_aaPlaceholder  = NULL;
    m_aaOperator     = NULL;
    m_aaTensorholder = NULL;

    m_PlaceholderDegree  = 0;
    m_OperatorDegree     = 0;
    m_TensorholderDegree = 0;

    m_aOptimizer          = NULL;
    m_PosOfObjectOperator = 0;
}

template<typename DTYPE> NeuralNetwork<DTYPE>::~NeuralNetwork() {
    std::cout << "NeuralNetwork<DTYPE>::~NeuralNetwork()" << '\n';
    this->Delete();
}

// template<typename DTYPE> int NeuralNetwork<DTYPE>::Alloc() {
// std::cout << "NeuralNetwork<DTYPE>::Alloc()" << '\n';
// return TRUE;
// }

template<typename DTYPE> void NeuralNetwork<DTYPE>::Delete() {
    std::cout << "NeuralNetwork<DTYPE>::Delete()" << '\n';

    if (m_aaPlaceholder) {
        for (int i = 0; i < m_PlaceholderDegree; i++) {
            delete m_aaPlaceholder[i];
        }
        delete[] m_aaPlaceholder;
    }

    if (m_aaOperator) {
        for (int i = 0; i < m_OperatorDegree; i++) {
            delete m_aaOperator[i];
        }
        delete[] m_aaOperator;
    }

    if (m_aaTensorholder) {
        for (int i = 0; i < m_TensorholderDegree; i++) {
            delete m_aaTensorholder[i];
        }
        delete[] m_aaTensorholder;
    }

    delete m_aOptimizer;
}

template<typename DTYPE> Placeholder<DTYPE> *NeuralNetwork<DTYPE>::AddPlaceholder(Placeholder<DTYPE> *pPlaceholder) {
    try {
        Placeholder<DTYPE> **temp = new Placeholder<DTYPE> *[m_PlaceholderDegree + 1];

        for (int i = 0; i < m_PlaceholderDegree; i++) temp[i] = m_aaPlaceholder[i];
        temp[m_PlaceholderDegree] = pPlaceholder;

        if (m_aaPlaceholder) {
            delete[] m_aaPlaceholder;
            m_aaPlaceholder = NULL;
        }

        m_aaPlaceholder = temp;
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return NULL;
    }

    m_PlaceholderDegree++;
    return pPlaceholder;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::AddOperator(Operator<DTYPE> *pOperator) {
    try {
        Operator<DTYPE> **temp = new Operator<DTYPE> *[m_OperatorDegree + 1];

        for (int i = 0; i < m_OperatorDegree; i++) temp[i] = m_aaOperator[i];
        temp[m_OperatorDegree] = pOperator;

        if (m_aaOperator) {
            delete[] m_aaOperator;
            m_aaOperator = NULL;
        }

        m_aaOperator = temp;
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return NULL;
    }

    m_OperatorDegree++;
    return pOperator;
}

template<typename DTYPE> Tensorholder<DTYPE> *NeuralNetwork<DTYPE>::AddTensorholder(Tensorholder<DTYPE> *pTensorholder) {
    try {
        Tensorholder<DTYPE> **temp = new Tensorholder<DTYPE> *[m_TensorholderDegree + 1];

        for (int i = 0; i < m_TensorholderDegree; i++) temp[i] = m_aaTensorholder[i];
        temp[m_TensorholderDegree] = pTensorholder;

        if (m_aaTensorholder) {
            delete[] m_aaTensorholder;
            m_aaTensorholder = NULL;
        }

        m_aaTensorholder = temp;
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return NULL;
    }

    m_TensorholderDegree++;
    return pTensorholder;
}

template<typename DTYPE> Optimizer<DTYPE> *NeuralNetwork<DTYPE>::SetOptimizer(Optimizer<DTYPE> *pOptimizer) {
    m_aOptimizer = pOptimizer;

    // for optimizer
    for (int i = 0; i < m_TensorholderDegree; i++) {
        m_aOptimizer->AddTrainableTensor(m_aaTensorholder[i]);
    }

    Operator<DTYPE> *objectoperator = m_aOptimizer->GetObjectOperator();

    for (int i = 0; i < m_OperatorDegree; i++) {
        if (m_aaOperator[i] == objectoperator) {
            m_PosOfObjectOperator = i;
        }
    }

    return m_aOptimizer;
}

// ===========================================================================================

template<typename DTYPE> int NeuralNetwork<DTYPE>::ForwardPropagate() {

    for(int i = 0; i <= m_PosOfObjectOperator; i++){
        m_aaOperator[i]->ComputeForwardPropagate();
    }

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ForwardPropagate(Operator<DTYPE> *pEnd) {
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ForwardPropagate(Operator<DTYPE> *pStart, Operator<DTYPE> *pEnd) {
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::BackPropagate() {
    for(int i = m_PosOfObjectOperator; i >= 0; i--){
        m_aaOperator[i]->ComputeBackPropagate();
    }
    return TRUE;
}

// =========

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::Training() {
    this->ForwardPropagate();
    this->BackPropagate();
    m_aOptimizer->UpdateVariable();

    return m_aOptimizer->GetObjectOperator();
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::Testing() {
    this->ForwardPropagate();

    return m_aOptimizer->GetObjectOperator();
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::Testing(Operator<DTYPE> *pEnd) {
    this->ForwardPropagate(pEnd);

    return pEnd;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::Testing(Operator<DTYPE> *pStart, Operator<DTYPE> *pEnd) {
    this->ForwardPropagate(pStart, pEnd);

    return pEnd;
}

// =========

template<typename DTYPE> int NeuralNetwork<DTYPE>::CreateGraph() {
    // in this part, we can check dependency between operator

    return TRUE;
}
