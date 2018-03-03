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

    m_aObjective = NULL;
    m_aOptimizer = NULL;
}

template<typename DTYPE> NeuralNetwork<DTYPE>::~NeuralNetwork() {
    std::cout << "NeuralNetwork<DTYPE>::~NeuralNetwork()" << '\n';
    this->Delete();
}

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

    if(m_aObjective){
        delete m_aObjective;
        m_aObjective = NULL;
    }

    if(m_aOptimizer){
        delete m_aOptimizer;
        m_aOptimizer = NULL;
    }
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

template<typename DTYPE> Objective<DTYPE> *NeuralNetwork<DTYPE>::SetObjective(Objective<DTYPE> *pObjective) {
    m_aObjective = pObjective;
    return pObjective;
}

template<typename DTYPE> Optimizer<DTYPE> *NeuralNetwork<DTYPE>::SetOptimizer(Optimizer<DTYPE> *pOptimizer) {
    m_aOptimizer = pOptimizer;
    m_aOptimizer->SetTrainableTensorDegree(m_TensorholderDegree);
    return pOptimizer;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::FeedData(int numOfPlaceholder, ...) {
    va_list ap;

    va_start(ap, numOfPlaceholder);

    // need to check compare between pRank value and number of another parameter
    for (int i = 0; i < numOfPlaceholder; i++) {
        // need to check whether int or not
        m_aaPlaceholder[i]->SetTensor(va_arg(ap, Tensor<DTYPE> *));
    }
    va_end(ap);

    return TRUE;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::GetResultOperator() {
    return m_aaOperator[m_OperatorDegree - 1];
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::GetResult() {
    return m_aaOperator[m_OperatorDegree - 1];
}

template<typename DTYPE> Tensorholder<DTYPE> **NeuralNetwork<DTYPE>::GetTensorholder() {
    return m_aaTensorholder;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::GetTensorholderDegree() {
    return m_TensorholderDegree;
}

template<typename DTYPE> Objective<DTYPE> *NeuralNetwork<DTYPE>::GetObjective() {
    return m_aObjective;
}

template<typename DTYPE> Optimizer<DTYPE> *NeuralNetwork<DTYPE>::GetOptimizer() {
    return m_aOptimizer;
}

template<typename DTYPE> float NeuralNetwork<DTYPE>::GetAccuracy() {
    Operator<DTYPE> *result
        = GetResultOperator();
    Operator<DTYPE> *label = m_aObjective->GetLabel();

    int batch = label->GetResult()->GetBatchSize();

    return (float)temp::Accuracy(result->GetResult(), label->GetResult(), batch);
}

template<typename DTYPE> float NeuralNetwork<DTYPE>::GetLoss() {
    float avg_loss = 0.f;

    int batch = m_aObjective->GetResult()->GetBatchSize();

    for (int k = 0; k < batch; k++) {
        avg_loss += (*m_aObjective)[k] / batch;
    }

    return avg_loss;
}

// ===========================================================================================

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::ForwardPropagate() {
    for (int i = 0; i < m_OperatorDegree; i++) {
        m_aaOperator[i]->ComputeForwardPropagate();
    }

    return m_aaOperator[m_OperatorDegree - 1];
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ForwardPropagate(Operator<DTYPE> *pEnd) {
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ForwardPropagate(Operator<DTYPE> *pStart, Operator<DTYPE> *pEnd) {
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::BackPropagate() {
    for (int i = m_OperatorDegree - 1; i >= 0; i--) {
        m_aaOperator[i]->ComputeBackPropagate();
    }
    return TRUE;
}

// =========

template<typename DTYPE> int NeuralNetwork<DTYPE>::Training() {
    ForwardPropagate();
    m_aObjective->ForwardPropagate();

    m_aObjective->BackPropagate();
    BackPropagate();

    m_aOptimizer->UpdateVariable();

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Testing() {
    ForwardPropagate();
    m_aObjective->ForwardPropagate();

    return TRUE;
}

// =========

template<typename DTYPE> int NeuralNetwork<DTYPE>::CreateGraph() {
    // in this part, we can check dependency between operator

    return TRUE;
}
