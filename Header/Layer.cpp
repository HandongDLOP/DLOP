#include "Layer.h"

template class Layer<int>;
template class Layer<float>;
template class Layer<double>;

//////////////////////////////////////////////////////////////////////////////// for private method

template<typename DTYPE> int Layer<DTYPE>::Alloc() {
    m_aaOperator = new Container<Operator<DTYPE> *>();
    return TRUE;
}

template<typename DTYPE> void Layer<DTYPE>::Delete() {
    #ifdef __DEBUG__
    std::cout << "Layer<DTYPE>::Delete()" << '\n';
    #endif  // __DEBUG__

    if (m_aaOperator) {
        Operator<DTYPE> **OperatorContainer = m_aaOperator->GetRawData();

        for (int i = 0; i < m_numOfOperator; i++) {
            delete OperatorContainer[i];
            OperatorContainer[i] = NULL;
        }
        delete m_aaOperator;
        m_aaOperator = NULL;
    }
}

//////////////////////////////////////////////////////////////////////////////// for public method

template<typename DTYPE> Layer<DTYPE>::Layer(std::string pName) : Operator<DTYPE>(pName) {
    #ifdef __DEBUG__
    std::cout << "Layer<DTYPE>::Layer()" << '\n';
    #endif  // __DEBUG__
    m_aaOperator = NULL;

    m_numOfOperator = 0;

    Alloc();
}

template<typename DTYPE> Layer<DTYPE>::~Layer() {
    #ifdef __DEBUG__
    std::cout << "Layer<DTYPE>::~Layer()" << '\n';
    #endif  // __DEBUG__

    this->Delete();
}

template<typename DTYPE> Operator<DTYPE> *Layer<DTYPE>::AddOperator(Operator<DTYPE> *pOperator) {
    int pNumOfParameter = pOperator->GetNumOfParameter();

    m_aaOperator->Push(pOperator);
    m_numOfOperator++;

    for (int i = 0; i < pNumOfParameter; i++) {
        this->AddParameter(pOperator->PopParameter());
    }

    return pOperator;
}

template<typename DTYPE> Container<Operator<DTYPE> *> *Layer<DTYPE>::GetOperatorContainer() {
    return m_aaOperator;
}

template<typename DTYPE> int Layer<DTYPE>::GetNumOfOperator() {
    return m_numOfOperator;
}

template<typename DTYPE> Operator<DTYPE> **Layer<DTYPE>::GetOutput() {
    return m_aaOperator->GetLast()->GetOutput();
}

template<typename DTYPE> Container<Operator<DTYPE> *> *Layer<DTYPE>::GetOutputContainer() {
    return m_aaOperator->GetLast()->GetOutputContainer();
}

template<typename DTYPE> Operator<DTYPE> **Layer<DTYPE>::GetInput() {
    return (*m_aaOperator)[0]->GetInput();
}

template<typename DTYPE> Container<Operator<DTYPE> *> *Layer<DTYPE>::GetInputContainer() {
    return (*m_aaOperator)[0]->GetInputContainer();
}

template<typename DTYPE> Tensor<DTYPE> *Layer<DTYPE>::GetResult() const {
    return m_aaOperator->GetLast()->GetResult();
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Layer<DTYPE>::GetResultContainer() {
    return m_aaOperator->GetLast()->GetResultContainer();
}

template<typename DTYPE> Tensor<DTYPE> *Layer<DTYPE>::GetGradient() const {
    return m_aaOperator->GetLast()->GetGradient();
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Layer<DTYPE>::GetGradientContainer() {
    return m_aaOperator->GetLast()->GetGradientContainer();
}

template<typename DTYPE> Tensor<DTYPE> *Layer<DTYPE>::GetDelta() const {
    return m_aaOperator->GetLast()->GetDelta();
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Layer<DTYPE>::GetDeltaContainer() {
    return m_aaOperator->GetLast()->GetDeltaContainer();
}

template<typename DTYPE> Operator<DTYPE> *Layer<DTYPE>::GetLastOperator() {
    return m_aaOperator->GetLast();
}

template<typename DTYPE> Operator<DTYPE> *Layer<DTYPE>::PopOperator() {
    m_numOfOperator--;
    return m_aaOperator->Pop();
}

template<typename DTYPE> int Layer<DTYPE>::ForwardPropagate(int pTime, int pThreadNum) {
    for (int i = 0; i < m_numOfOperator; i++) {
        (*m_aaOperator)[i]->ForwardPropagate(pTime, pThreadNum);
    }
    return TRUE;
}

template<typename DTYPE> int Layer<DTYPE>::BackPropagate(int pTime, int pThreadNum) {
    for (int i = m_numOfOperator - 1; i >= 0; i--) {
        (*m_aaOperator)[i]->BackPropagate(pTime, pThreadNum);
    }
    return TRUE;
}

template<typename DTYPE> int Layer<DTYPE>::ResetResult() {
    for (int i = m_numOfOperator - 1; i >= 0; i--) {
        (*m_aaOperator)[i]->ResetResult();
    }
    return TRUE;
}

template<typename DTYPE> int Layer<DTYPE>::ResetGradient() {
    for (int i = m_numOfOperator - 1; i >= 0; i--) {
        (*m_aaOperator)[i]->ResetGradient();
    }
    return TRUE;
}

template<typename DTYPE> void Layer<DTYPE>::PrintInformation() {
    std::cout << this->GetName() << " : ";
    std::cout << this->GetResult()->GetShape() << '\n';

    for (int i = 0; i < m_numOfOperator; i++) {
        std::cout << "-- ";
        (*m_aaOperator)[i]->PrintInformation();
    }
}

template<typename DTYPE> void Layer<DTYPE>::SetDeviceCPU() {
    this->SetDevice(CPU);

    for (int i = 0; i < m_numOfOperator; i++) {
        (*m_aaOperator)[i]->SetDeviceCPU();
    }
}

template<typename DTYPE> void Layer<DTYPE>::SetDeviceCPU(int pNumOfThread) {
    this->SetDevice(CPU);
    this->SetNumOfThread(pNumOfThread);

    for (int i = 0; i < m_numOfOperator; i++) {
        (*m_aaOperator)[i]->SetDeviceCPU(pNumOfThread);
    }
}

#ifdef __CUDNN__

template<typename DTYPE> void Layer<DTYPE>::SetDeviceGPU() {
    this->SetDevice(GPU);

    for (int i = 0; i < m_numOfOperator; i++) {
        (*m_aaOperator)[i]->SetDeviceGPU();
    }
}

template<typename DTYPE> void Layer<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle) {
    this->SetDevice(GPU);
    this->SetCudnnHandle(pCudnnHandle);

    for (int i = 0; i < m_numOfOperator; i++) {
        (*m_aaOperator)[i]->SetDeviceGPU(pCudnnHandle);
    }
}

template<typename DTYPE> int Layer<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    for (int i = 0; i < m_numOfOperator; i++) {
        (*m_aaOperator)[i]->ForwardPropagateOnGPU(pTime);
    }
    return TRUE;
}

template<typename DTYPE> int Layer<DTYPE>::BackPropagateOnGPU(int pTime) {
    for (int i = m_numOfOperator - 1; i >= 0; i--) {
        (*m_aaOperator)[i]->BackPropagateOnGPU(pTime);
    }
    return TRUE;
}

#endif  // if __CUDNN__
