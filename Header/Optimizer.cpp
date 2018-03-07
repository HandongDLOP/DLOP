#include "Optimizer.h"

template class Optimizer<int>;
template class Optimizer<float>;
template class Optimizer<double>;

template<typename DTYPE> Optimizer<DTYPE>::Optimizer(Tensorholder<DTYPE> **pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection) {
    std::cout << "Optimizer::Optimizer(Operator<DTYPE> *, float, OptimizeDirection)" << '\n';
    m_LearningRate          = 0.f;
    m_OptimizeDirection     = 1;
    m_ppTrainableTensors    = NULL;
    m_TrainableTensorDegree = 0;

    Alloc(pTrainableTensors, pLearningRate, pOptimizeDirection);
}

template<typename DTYPE> Optimizer<DTYPE>::~Optimizer() {
    std::cout << "Optimizer::~Optimizer()" << '\n';

    this->Delete();
}

template<typename DTYPE> int Optimizer<DTYPE>::Alloc(Tensorholder<DTYPE> **pTrainableTensors, float pLearningRate, OptimizeDirection pOptimizeDirection) {
    m_ppTrainableTensors = pTrainableTensors;

    m_LearningRate = pLearningRate;

    if (pOptimizeDirection == MAXIMIZE) m_OptimizeDirection = 1;
    else if (pOptimizeDirection == MINIMIZE) m_OptimizeDirection = -1;

    return TRUE;
}

template<typename DTYPE> int Optimizer<DTYPE>::Delete() {
    // delete m_ppTrainableTensors;

    return TRUE;
}

template<typename DTYPE> int Optimizer<DTYPE>::UpdateVariable() {
    for (int i = 0; i < m_TrainableTensorDegree; i++) {
        // UpdateVariable(m_aTrainableData[i]->Data, m_aTrainableData[i]->Gradient);
        UpdateVariable(m_ppTrainableTensors[i]);
    }
    return TRUE;
}

template<typename DTYPE> void Optimizer<DTYPE>::SetLearningRate(float pLearningRate) {
    m_LearningRate = pLearningRate;
}

template<typename DTYPE> void Optimizer<DTYPE>::SetTrainableTensorDegree(int pTrainableTensorDegree) {
    m_TrainableTensorDegree = pTrainableTensorDegree;
}

template<typename DTYPE> float Optimizer<DTYPE>::GetLearningRate()  const {
    return m_LearningRate;
}

template<typename DTYPE> int Optimizer<DTYPE>::GetOptimizeDirection() const {
    return m_OptimizeDirection;
}
