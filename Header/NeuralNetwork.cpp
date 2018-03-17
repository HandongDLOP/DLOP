#include "NeuralNetwork.h"

template class NeuralNetwork<int>;
template class NeuralNetwork<float>;
template class NeuralNetwork<double>;

template<typename DTYPE> NeuralNetwork<DTYPE>::NeuralNetwork() {
    std::cout << "NeuralNetwork<DTYPE>::NeuralNetwork()" << '\n';

#if __CUDNN__
    checkCUDNN(cudnnCreate(&m_cudnnHandle));
#endif  // if __CUDNN__

    m_aaOperator     = NULL;
    m_aaTensorholder = NULL;
    m_aaLayer        = NULL;

    m_OperatorDegree     = 0;
    m_TensorholderDegree = 0;

    m_aObjective = NULL;
    m_aOptimizer = NULL;

    Alloc();
}

template<typename DTYPE> NeuralNetwork<DTYPE>::~NeuralNetwork() {
    std::cout << "NeuralNetwork<DTYPE>::~NeuralNetwork()" << '\n';

#if __CUDNN__
    checkCudaErrors(cudaDeviceSynchronize());
    checkCUDNN(cudnnDestroy(m_cudnnHandle));
#endif  // if __CUDNN__

    this->Delete();
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Alloc() {
    m_aaOperator     = new Container<Operator<DTYPE> *>();
    m_aaTensorholder = new Container<Tensorholder<DTYPE> *>();
    m_aaLayer        = new Container<Layer<DTYPE> *>();

    return TRUE;
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::Delete() {
    std::cout << "NeuralNetwork<DTYPE>::Delete()" << '\n';
    int size = 0;

    if (m_aaOperator) {
        size = m_aaOperator->GetSize();
        Operator<DTYPE> **OperatorContainer = m_aaOperator->GetRawData();

        for (int i = 0; i < size; i++) {
            if ((*m_aaOperator)[i]) {
                delete OperatorContainer[i];
                OperatorContainer[i] = NULL;
            }
        }
        delete m_aaOperator;
        m_aaOperator = NULL;
    }

    if (m_aaTensorholder) {
        size = m_aaTensorholder->GetSize();
        Tensorholder<DTYPE> **TensorholderContainer = m_aaTensorholder->GetRawData();

        for (int i = 0; i < size; i++) {
            if ((*m_aaTensorholder)[i]) {
                delete TensorholderContainer[i];
                TensorholderContainer[i] = NULL;
            }
        }
        delete m_aaTensorholder;
        m_aaTensorholder = NULL;
    }

    if (m_aaLayer) {
        size = m_aaLayer->GetSize();
        Layer<DTYPE> **LayerContainer = m_aaLayer->GetRawData();

        for (int i = 0; i < size; i++) {
            if ((*m_aaLayer)[i]) {
                delete LayerContainer[i];
                LayerContainer[i] = NULL;
            }
        }
        delete m_aaLayer;
        m_aaLayer = NULL;
    }

    if (m_aObjective) {
        delete m_aObjective;
        m_aObjective = NULL;
    }

    if (m_aOptimizer) {
        delete m_aOptimizer;
        m_aOptimizer = NULL;
    }
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::AddOperator(Operator<DTYPE> *pOperator) {
#if __CUDNN__
    pOperator->SetCudnnHandle(m_cudnnHandle);
#endif  // if __CUDNN__
    m_aaOperator->Push(pOperator);
    m_OperatorDegree++;
    return pOperator;
}

template<typename DTYPE> Tensorholder<DTYPE> *NeuralNetwork<DTYPE>::AddTensorholder(Tensorholder<DTYPE> *pTensorholder) {
    m_aaTensorholder->Push(pTensorholder);
    m_TensorholderDegree++;
    return pTensorholder;
}

template<typename DTYPE> Tensorholder<DTYPE> *NeuralNetwork<DTYPE>::AddParameter(Tensorholder<DTYPE> *pTensorholder) {
    m_aaTensorholder->Push(pTensorholder);
    m_TensorholderDegree++;
    return pTensorholder;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::AddLayer(Layer<DTYPE> *pLayer) {
    int pNumOfOperator  = pLayer->GetNumOfOperator();
    int pNumOfParameter = pLayer->GetNumOfParameter();

    for (int i = 0; i < pNumOfOperator; i++) {
        m_aaOperator->Push(pLayer->PopOperator());
        m_OperatorDegree++;
#if __CUDNN__
        (*m_aaOperator)[m_OperatorDegree - 1]->SetCudnnHandle(m_cudnnHandle);
#endif  // if __CUDNN__
    }

    for (int i = 0; i < pNumOfParameter; i++) {
        m_aaTensorholder->Push(pLayer->PopParameter());
        m_TensorholderDegree++;
    }

    m_aaLayer->Push(pLayer);

    return m_aaOperator->GetLast();
}

template<typename DTYPE> Objective<DTYPE> *NeuralNetwork<DTYPE>::SetObjective(Objective<DTYPE> *pObjective) {
    m_aObjective = pObjective;
    return pObjective;
}

template<typename DTYPE> Optimizer<DTYPE> *NeuralNetwork<DTYPE>::SetOptimizer(Optimizer<DTYPE> *pOptimizer) {
    m_aOptimizer = pOptimizer;
    return pOptimizer;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::GetResultOperator() {
    return m_aaOperator->GetLast();
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::GetResult() {
    return m_aaOperator->GetLast();
}

template<typename DTYPE> Container<Tensorholder<DTYPE> *> *NeuralNetwork<DTYPE>::GetTensorholder() {
    return m_aaTensorholder;
}

template<typename DTYPE> Container<Tensorholder<DTYPE> *> *NeuralNetwork<DTYPE>::GetParameter() {
    return m_aaTensorholder;
}

template<typename DTYPE> Objective<DTYPE> *NeuralNetwork<DTYPE>::GetObjective() {
    return m_aObjective;
}

template<typename DTYPE> Optimizer<DTYPE> *NeuralNetwork<DTYPE>::GetOptimizer() {
    return m_aOptimizer;
}

template<typename DTYPE> float NeuralNetwork<DTYPE>::GetAccuracy() {
    Operator<DTYPE> *result = GetResultOperator();
    Operator<DTYPE> *label  = m_aObjective->GetLabel();

    int batch = label->GetResult()->GetBatchSize();

    Tensor<DTYPE> *pred = result->GetResult();
    Tensor<DTYPE> *ans  = label->GetResult();

    float accuracy = 0.f;

    int pred_index = 0;
    int ans_index  = 0;

    for (int ba = 0; ba < batch; ba++) {
        pred_index = GetMaxIndex(pred, ba, 10);
        ans_index  = GetMaxIndex(ans, ba, 10);

        if (pred_index == ans_index) {
            accuracy += 1.f;
        }
    }

    return (float)(accuracy / batch);
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::GetMaxIndex(Tensor<DTYPE> *data, int ba, int numOfClass) {
    int   index = 0;
    DTYPE max   = (*data)[ba * numOfClass];
    int   start = ba * numOfClass;
    int   end   = ba * numOfClass + numOfClass;

    for (int dim = start + 1; dim < end; dim++) {
        if ((*data)[dim] > max) {
            max   = (*data)[dim];
            index = dim - start;
        }
    }

    return index;
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

template<typename DTYPE> int NeuralNetwork<DTYPE>::ForwardPropagate() {
    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->ComputeForwardPropagate();
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
    for (int i = m_OperatorDegree - 1; i >= 0; i--) {
        (*m_aaOperator)[i]->ComputeBackPropagate();
    }
    return TRUE;
}

// =========

template<typename DTYPE> int NeuralNetwork<DTYPE>::Training() {
    this->ResetOperatorResult();
    this->ResetOperatorGradient();
    this->ResetObjectiveResult();
    this->ResetObjectiveGradient();

    this->ForwardPropagate();
    m_aObjective->ComputeForwardPropagate();

    m_aObjective->ComputeBackPropagate();
    this->BackPropagate();

    m_aOptimizer->UpdateVariable();

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Testing() {
    this->ResetOperatorResult();
    this->ResetObjectiveResult();
    ForwardPropagate();
    m_aObjective->ComputeForwardPropagate();

    return TRUE;
}

// =========

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeTraining() {
    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->SetModeTraining();
    }
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeAccumulating() {
    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->SetModeAccumulating();
    }
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeInferencing() {
    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->SetModeInferencing();
    }
}

// =========

template<typename DTYPE> int NeuralNetwork<DTYPE>::CreateGraph() {
    // in this part, we can check dependency between operator

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::PrintGraphShape() {
    for (int i = 0; i < m_OperatorDegree; i++) {
        std::cout << (*m_aaOperator)[i]->GetName() << '\n';
        std::cout << (*m_aaOperator)[i]->GetResult()->GetShape() << '\n';
    }

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetOperatorResult() {
    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->ResetResult();
    }
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetOperatorGradient() {
    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->ResetGradient();
    }
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetObjectiveResult() {
    m_aObjective->ResetResult();
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetObjectiveGradient() {
    m_aObjective->ResetGradient();
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetParameterGradient() {
    m_aOptimizer->ResetParameterGradient();
    return TRUE;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::SerchOperator(std::string pName) {
    std::string name = "NULL";

    for (int i = 0; i < m_OperatorDegree; i++) {
        name = (*m_aaOperator)[i]->GetName();
        if(name == pName) return (*m_aaOperator)[i];
    }

    return NULL;
}
