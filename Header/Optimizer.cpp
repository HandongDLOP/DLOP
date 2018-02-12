#include "Optimizer.h"

template class Optimizer<int>;
template class Optimizer<float>;
template class Optimizer<double>;

template<typename DTYPE> Optimizer<DTYPE>::Optimizer(NeuralNetwork<DTYPE> *pNeuralNetwork, float pLearningRate, OptimizeDirection pOptimizeDirection) {
    std::cout << "Optimizer::Optimizer(Operator<DTYPE> *, float, OptimizeDirection)" << '\n';
    m_pNeuralNetwork = NULL;
    m_LearningRate = 0.f;
    m_OptimizeDirection = 1;
    m_apTrainableTensor = NULL;
    m_TrainableTensorDegree = 0;

    Alloc(pNeuralNetwork, pLearningRate, pOptimizeDirection);
}

template<typename DTYPE> Optimizer<DTYPE>::~Optimizer() {
    std::cout << "Optimizer::~Optimizer()" << '\n';

    this->Delete();
}

template<typename DTYPE> int Optimizer<DTYPE>::Alloc(NeuralNetwork<DTYPE> *pNeuralNetwork, float pLearningRate, OptimizeDirection pOptimizeDirection) {
    m_pNeuralNetwork = pNeuralNetwork;

    this->AddTrainableTensor(pNeuralNetwork);

    m_LearningRate = pLearningRate;

    if (pOptimizeDirection == MAXIMIZE) m_OptimizeDirection = 1;
    else if (pOptimizeDirection == MINIMIZE) m_OptimizeDirection = -1;

    return TRUE;
}

template<typename DTYPE> int Optimizer<DTYPE>::Delete() {
    delete m_apTrainableTensor;

    return TRUE;
}

template<typename DTYPE> int Optimizer<DTYPE>::AddTrainableTensor(NeuralNetwork<DTYPE> *pNeuralNetwork) {
    Tensorholder<DTYPE> ** pTensorholders = pNeuralNetwork->GetTensorholder();
    int pTensorholderDegree = pNeuralNetwork->GetTensorholderDegree();

    for (int i = 0; i < pTensorholderDegree; i++) {
        this->AddTrainableTensor(pTensorholders[i]);
    }

    return TRUE;
}

template<typename DTYPE> int Optimizer<DTYPE>::AddTrainableTensor(Operator<DTYPE> *pTrainableTensor) {
    try {
        Operator<DTYPE> **temp = new Operator<DTYPE> *[m_TrainableTensorDegree + 1];

        for (int i = 0; i < m_TrainableTensorDegree; i++) temp[i] = m_apTrainableTensor[i];
        temp[m_TrainableTensorDegree] = pTrainableTensor;

        if (m_apTrainableTensor) {
            delete[] m_apTrainableTensor;
            m_apTrainableTensor = NULL;
        }

        m_apTrainableTensor = temp;
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }
    m_TrainableTensorDegree++;

    return TRUE;
}

template<typename DTYPE> int Optimizer<DTYPE>::UpdateVariable() {
    for (int i = 0; i < m_TrainableTensorDegree; i++) {
        // UpdateVariable(m_aTrainableData[i]->Data, m_aTrainableData[i]->Gradient);
        UpdateVariable(m_apTrainableTensor[i]);
    }
    return TRUE;
}

template<typename DTYPE> void Optimizer<DTYPE>::SetLearningRate(float pLearningRate) {
    m_LearningRate = pLearningRate;
}

template<typename DTYPE> float Optimizer<DTYPE>::GetLearningRate()  const {
    return m_LearningRate;
}

template<typename DTYPE> int Optimizer<DTYPE>::GetOptimizeDirection() const {
    return m_OptimizeDirection;
}
