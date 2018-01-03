#include "Optimizer.h"

template class Optimizer<int>;
template class Optimizer<float>;
template class Optimizer<double>;

template<typename DTYPE> Optimizer<DTYPE>::Optimizer(Operator<DTYPE> *pObjectOperator, float pLearningRate, OptimizeDirection pOptimizeDirection) {
    std::cout << "Optimizer::Optimizer(Operator<DTYPE> *, float, OptimizeDirection)" << '\n';
    m_pObjectOperator       = NULL;
    m_LearningRate          = 0.f;
    m_OptimizeDirection     = 1;
    m_apTrainableTensor     = NULL;
    m_TrainableTensorDegree = 0;

    Alloc(pObjectOperator, pLearningRate, pOptimizeDirection);
}

template<typename DTYPE> Optimizer<DTYPE>::~Optimizer() {
    std::cout << "Optimizer::~Optimizer()" << '\n';

    Delete();
}

template<typename DTYPE> int Optimizer<DTYPE>::Alloc(Operator<DTYPE> *pObjectOperator, float pLearningRate, OptimizeDirection pOptimizeDirection) {
    m_pObjectOperator = pObjectOperator;
    m_LearningRate    = pLearningRate;

    if (pOptimizeDirection == MAXIMIZE) m_OptimizeDirection = 1;
    else if (pOptimizeDirection == MINIMIZE) m_OptimizeDirection = -1;

    return TRUE;
}

template<typename DTYPE> int Optimizer<DTYPE>::Delete() {
    for (int i = 0; i < m_TrainableTensorDegree; i++) {
        delete m_apTrainableTensor[i];
    }
    delete m_apTrainableTensor;

    return TRUE;
}

template<typename DTYPE> int Optimizer<DTYPE>::AddTrainableData(Tensorholder<DTYPE> *pTrainableTensor) {
    return this->AddTrainableTensor(pTrainableTensor);
}

template<typename DTYPE> int Optimizer<DTYPE>::AddTrainableTensor(Tensorholder<DTYPE> *pTrainableTensor) {
    try {
        Tensorholder<DTYPE> **temp = new Tensorholder<DTYPE> *[m_TrainableTensorDegree + 1];

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

template<typename DTYPE> Operator<DTYPE> *Optimizer<DTYPE>::GetObjectOperator() const {
    return m_pObjectOperator;
}

template<typename DTYPE> float Optimizer<DTYPE>::GetLearningRate()  const {
    return m_LearningRate;
}

template<typename DTYPE> int Optimizer<DTYPE>::GetOptimizeDirection() const {
    return m_OptimizeDirection;
}
