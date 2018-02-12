#include "Model.h"

template class Model<int>;
template class Model<float>;
template class Model<double>;

template<typename DTYPE> Model<DTYPE>::Model() {
    std::cout << "Model<DTYPE>::Model()" << '\n';
    m_aaNeuralNetworks = NULL;
    m_aObjective = NULL;
    m_aOptimizer = NULL;
    m_NeuralNetworkDegree = 0;
}

template<typename DTYPE> Model<DTYPE>::~Model() {
    std::cout << "Model<DTYPE>::~Model()" << '\n';
    this->Delete();
}

template<typename DTYPE> int Model<DTYPE>::Alloc() {
    std::cout << "Model<DTYPE>::Alloc()" << '\n';
    return TRUE;
}

template<typename DTYPE> void Model<DTYPE>::Delete() {
    std::cout << "Model<DTYPE>::Delete()" << '\n';
}

template<typename DTYPE> NeuralNetwork<DTYPE> *Model<DTYPE>::AddNeuralNetwork(NeuralNetwork<DTYPE> *pNeuralNetwork) {
    try {
        NeuralNetwork<DTYPE> **temp = new NeuralNetwork<DTYPE> *[m_NeuralNetworkDegree + 1];

        for (int i = 0; i < m_NeuralNetworkDegree; i++) temp[i] = m_aaNeuralNetworks[i];
        temp[m_NeuralNetworkDegree] = pNeuralNetwork;

        if (m_aaNeuralNetworks) {
            delete[] m_aaNeuralNetworks;
            m_aaNeuralNetworks = NULL;
        }

        m_aaNeuralNetworks = temp;
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return NULL;
    }
    m_NeuralNetworkDegree++;

    return pNeuralNetwork;
}

template<typename DTYPE> Objective<DTYPE> *Model<DTYPE>::SetObjective(Objective<DTYPE> *pObjective) {
    m_aObjective = pObjective;
    return pObjective;
}

template<typename DTYPE> Optimizer<DTYPE> *Model<DTYPE>::SetOptimizer(Optimizer<DTYPE> *pOptimizer) {
    m_aOptimizer = pOptimizer;
    return pOptimizer;
}

template<typename DTYPE> int Model<DTYPE>::Training() {
    for (int i = 0; i < m_NeuralNetworkDegree; i++) m_aaNeuralNetworks[i]->ForwardPropagate();
    m_aObjective->ForwardPropagate();

    m_aObjective->BackPropagate();
    for (int i = 0; i < m_NeuralNetworkDegree; i++) m_aaNeuralNetworks[i]->BackPropagate();

    m_aOptimizer->UpdateVariable();

    return TRUE;
}

template<typename DTYPE> int Model<DTYPE>::Testing() {
    for (int i = 0; i < m_NeuralNetworkDegree; i++) m_aaNeuralNetworks[i]->ForwardPropagate();
    m_aObjective->ForwardPropagate();

    return TRUE;
}

template<typename DTYPE> NeuralNetwork<DTYPE> **Model<DTYPE>::GetNeuralNetworks() {
    return m_aaNeuralNetworks;
}

template<typename DTYPE> Objective<DTYPE> *Model<DTYPE>::GetObjective() {
    return m_aObjective;
}

template<typename DTYPE> Optimizer<DTYPE> *Model<DTYPE>::GetOptimizer() {
    return m_aOptimizer;
}

template<typename DTYPE> float Model<DTYPE>::GetAccuracy() {
    Operator<DTYPE> * result = m_aaNeuralNetworks[m_NeuralNetworkDegree-1]->GetResultOperator();
    Operator<DTYPE> * label = m_aObjective->GetLabel();

    int batch = label->GetResult()->GetBatchSize();

    return (float)temp::Accuracy(result->GetResult(), label->GetResult(), batch);
}

template<typename DTYPE> float Model<DTYPE>::GetLoss() {
    float avg_loss = 0.f;

    int batch = m_aObjective->GetResult()->GetBatchSize();

    for (int k = 0; k < batch; k++) {
        avg_loss += (*m_aObjective)[k] / batch;
    }

    return avg_loss;
}
