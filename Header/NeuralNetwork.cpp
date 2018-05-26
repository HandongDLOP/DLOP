#include "NeuralNetwork.h"

template class NeuralNetwork<int>;
template class NeuralNetwork<float>;
template class NeuralNetwork<double>;

//////////////////////////////////////////////////////////////////////////////// for private method

template<typename DTYPE> int NeuralNetwork<DTYPE>::Alloc() {
    m_aaOperator  = new Container<Operator<DTYPE> *>();
    m_apInput     = new Container<Operator<DTYPE> *>();
    m_aaParameter = new Container<Operator<DTYPE> *>();
    return TRUE;
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::Delete() {
    #ifdef __DEBUG__
    std::cout << "NeuralNetwork<DTYPE>::Delete()" << '\n';
    #endif  // __DEBUG__
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

    if (m_apInput) {
        delete m_apInput;
        m_apInput = NULL;
    }

    if (m_aaParameter) {
        size = m_aaParameter->GetSize();
        Operator<DTYPE> **ParameterContainer = m_aaParameter->GetRawData();

        for (int i = 0; i < size; i++) {
            if ((*m_aaParameter)[i]) {
                delete ParameterContainer[i];
                ParameterContainer[i] = NULL;
            }
        }
        delete m_aaParameter;
        m_aaParameter = NULL;
    }

    if (m_aLossFunction) {
        delete m_aLossFunction;
        m_aLossFunction = NULL;
    }

    if (m_aOptimizer) {
        delete m_aOptimizer;
        m_aOptimizer = NULL;
    }

#ifdef __CUDNN__
    this->DeleteOnGPU();
#endif  // if __CUDNN__
}

#ifdef __CUDNN__
template<typename DTYPE> int NeuralNetwork<DTYPE>::AllocOnGPU() {
    // checkCudaErrors(cudaSetDevice(2));
    checkCUDNN(cudnnCreate(&m_cudnnHandle));
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::DeleteOnGPU() {
    checkCudaErrors(cudaThreadSynchronize());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCUDNN(cudnnDestroy(m_cudnnHandle));
}

#endif  // if __CUDNN__

//////////////////////////////////////////////////////////////////////////////// for public method


template<typename DTYPE> NeuralNetwork<DTYPE>::NeuralNetwork() {
    #ifdef __DEBUG__
    std::cout << "NeuralNetwork<DTYPE>::NeuralNetwork()" << '\n';
    #endif  // __DEBUG__

    m_aaOperator  = NULL;
    m_apInput     = NULL;
    m_aaParameter = NULL;

    m_OperatorDegree  = 0;
    m_ParameterDegree = 0;

    m_aLossFunction = NULL;
    m_aOptimizer    = NULL;

    m_Device      = CPU;
    m_numOfThread = 1;

#ifdef __CUDNN__
    m_cudnnHandle = NULL;
#endif  // if __CUDNN__

    Alloc();
}

template<typename DTYPE> NeuralNetwork<DTYPE>::~NeuralNetwork() {
    #ifdef __DEBUG__
    std::cout << "NeuralNetwork<DTYPE>::~NeuralNetwork()" << '\n';
    #endif  // __DEBUG__

    this->Delete();
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::SetInput(Operator<DTYPE> *pInput) {
    return pInput;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::AnalyseGraph(Operator<DTYPE> *pResultOperator) {
    Container<Operator<DTYPE> *> queue;
    queue.Push(pResultOperator);
    Operator<DTYPE> *out                 = NULL;
    Container<Operator<DTYPE> *> *nextOp = NULL;
    int numOfEdge                        = 0;

    while (queue.GetSize() > 0) {
        out = queue.Pop();

        nextOp    = out->GetInputContainer();
        numOfEdge = nextOp->GetSize();

        for (int i = 0; i < numOfEdge; i++) {
            queue.Push((*nextOp)[i]);
        }

        if (out->GetIsTensorholder()) {
            if(out->GetIsTrainable()) m_aaParameter->Push(out);
        } else {
            m_aaOperator->Push(out);
        }
    }

    m_aaOperator->Reverse();
    m_aaParameter->Reverse();

    m_OperatorDegree = m_aaOperator->GetSize();

    for (int i = 0; i < m_OperatorDegree; i++) {
        std::cout << (*m_aaOperator)[i]->GetName() << '\n';
    }

    m_ParameterDegree = m_aaParameter->GetSize();

    for (int i = 0; i < m_ParameterDegree; i++) {
        std::cout << (*m_aaParameter)[i]->GetName() << '\n';
    }

    return pResultOperator;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::AddOperator(Operator<DTYPE> *pOperator) {
    int pNumOfParameter = pOperator->GetNumOfParameter();

    m_aaOperator->Push(pOperator);
    m_OperatorDegree++;

    for (int i = 0; i < pNumOfParameter; i++) {
        m_aaParameter->Push(pOperator->PopParameter());
        m_ParameterDegree++;
    }

    return pOperator;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::AddParameter(Operator<DTYPE> *pParameter) {
    m_aaParameter->Push(pParameter);
    m_ParameterDegree++;
    return pParameter;
}

template<typename DTYPE> LossFunction<DTYPE> *NeuralNetwork<DTYPE>::SetLossFunction(LossFunction<DTYPE> *pLossFunction) {
    m_aLossFunction = pLossFunction;
    return pLossFunction;
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

template<typename DTYPE> Container<Operator<DTYPE> *> *NeuralNetwork<DTYPE>::GetOperatorContainer() {
    return m_aaOperator;
}

template<typename DTYPE> Container<Operator<DTYPE> *> *NeuralNetwork<DTYPE>::GetParameter() {
    return m_aaParameter;
}

template<typename DTYPE> LossFunction<DTYPE> *NeuralNetwork<DTYPE>::GetLossFunction() {
    return m_aLossFunction;
}

template<typename DTYPE> Optimizer<DTYPE> *NeuralNetwork<DTYPE>::GetOptimizer() {
    return m_aOptimizer;
}

template<typename DTYPE> float NeuralNetwork<DTYPE>::GetAccuracy() {
    Operator<DTYPE> *result = GetResultOperator();
    Operator<DTYPE> *label  = m_aLossFunction->GetLabel();

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

    int batch = m_aLossFunction->GetResult()->GetBatchSize();

    for (int k = 0; k < batch; k++) {
        avg_loss += (*m_aLossFunction)[k] / batch;
    }

    return avg_loss;
}

// ===========================================================================================

template<typename DTYPE> int NeuralNetwork<DTYPE>::ForwardPropagate(int pTime) {
    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->ForwardPropagate(pTime);
    }
    m_aLossFunction->ForwardPropagate(pTime);

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::BackPropagate(int pTime) {
    m_aLossFunction->BackPropagate(pTime);

    for (int i = m_OperatorDegree - 1; i >= 0; i--) {
        (*m_aaOperator)[i]->BackPropagate(pTime);
    }
    return TRUE;
}

template<typename DTYPE> void *NeuralNetwork<DTYPE>::ForwardPropagateForThread(void *param) {
    ThreadInfo *pThreadInfo = (ThreadInfo *)param;

    NeuralNetwork<DTYPE> *pNN = (NeuralNetwork<DTYPE> *)(pThreadInfo->m_NN);
    int pTime                 = 0;
    int pThreadNum            = pThreadInfo->m_threadNum;

    Container<Operator<DTYPE> *> *m_aaOperator = pNN->GetOperatorContainer();
    int m_OperatorDegree                       = m_aaOperator->GetSize();
    LossFunction<DTYPE> *m_aLossFunction       = pNN->GetLossFunction();

    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->ForwardPropagate(pTime, pThreadNum);
    }
    m_aLossFunction->ForwardPropagate(pTime, pThreadNum);
    return NULL;
}

template<typename DTYPE> void *NeuralNetwork<DTYPE>::BackPropagateForThread(void *param) {
    ThreadInfo *pThreadInfo = (ThreadInfo *)param;

    NeuralNetwork<DTYPE> *pNN = (NeuralNetwork<DTYPE> *)(pThreadInfo->m_NN);
    int pTime                 = 0;
    int pThreadNum            = pThreadInfo->m_threadNum;

    Container<Operator<DTYPE> *> *m_aaOperator = pNN->GetOperatorContainer();
    int m_OperatorDegree                       = m_aaOperator->GetSize();
    LossFunction<DTYPE> *m_aLossFunction       = pNN->GetLossFunction();

    m_aLossFunction->BackPropagate(pTime, pThreadNum);

    for (int i = m_OperatorDegree - 1; i >= 0; i--) {
        (*m_aaOperator)[i]->BackPropagate(pTime, pThreadNum);
    }
    return NULL;
}

#ifdef __CUDNN__
template<typename DTYPE> int NeuralNetwork<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->ForwardPropagateOnGPU(pTime);
    }
    m_aLossFunction->ForwardPropagateOnGPU(pTime);

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::BackPropagateOnGPU(int pTime) {
    m_aLossFunction->BackPropagateOnGPU(pTime);

    for (int i = m_OperatorDegree - 1; i >= 0; i--) {
        (*m_aaOperator)[i]->BackPropagateOnGPU(pTime);
    }
    return TRUE;
}

#endif  // __CUDNN__


// =========

template<typename DTYPE> int NeuralNetwork<DTYPE>::Training() {
    if ((m_Device == CPU) && (m_numOfThread > 1)) {
        this->TrainingOnMultiThread();
    } else if ((m_Device == CPU) && (m_numOfThread == 1)) {
        this->TrainingOnCPU();
    } else if (m_Device == GPU) {
        this->TrainingOnGPU();
    } else return FALSE;

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Testing() {
    if ((m_Device == CPU) && (m_numOfThread > 1)) {
        this->TestingOnMultiThread();
    } else if ((m_Device == CPU) && (m_numOfThread == 1)) {
        this->TestingOnCPU();
    } else if (m_Device == GPU) {
        this->TestingOnGPU();
    } else return FALSE;

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::TrainingOnCPU() {
    this->ResetOperatorResult();
    this->ResetOperatorGradient();
    this->ResetLossFunctionResult();
    this->ResetLossFunctionGradient();

    this->ForwardPropagate();
    this->BackPropagate();

    m_aOptimizer->UpdateVariable();

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::TestingOnCPU() {
    this->ResetOperatorResult();
    this->ResetLossFunctionResult();

    this->ForwardPropagate();
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::TrainingOnMultiThread() {
    this->ResetOperatorResult();
    this->ResetOperatorGradient();
    this->ResetLossFunctionResult();
    this->ResetLossFunctionGradient();

    pthread_t  *pThread     = (pthread_t *)malloc(sizeof(pthread_t) * m_numOfThread);
    ThreadInfo *pThreadInfo = (ThreadInfo *)malloc(sizeof(ThreadInfo) * m_numOfThread);

    for (int i = 0; i < m_numOfThread; i++) {
        pThreadInfo[i].m_NN        = (void *)this;
        pThreadInfo[i].m_threadNum = i;
        pthread_create(&(pThread[i]), NULL, ForwardPropagateForThread, (void *)&(pThreadInfo[i]));
    }

    for (int i = 0; i < m_numOfThread; i++) {
        pthread_join(pThread[i], NULL);
    }

    for (int i = 0; i < m_numOfThread; i++) {
        pthread_create(&(pThread[i]), NULL, BackPropagateForThread, (void *)&(pThreadInfo[i]));
    }

    for (int i = 0; i < m_numOfThread; i++) {
        pthread_join(pThread[i], NULL);
    }

    free(pThread);
    free(pThreadInfo);

    m_aOptimizer->UpdateVariable();

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::TestingOnMultiThread() {
    this->ResetOperatorResult();
    this->ResetLossFunctionResult();

    pthread_t  *pThread     = (pthread_t *)malloc(sizeof(pthread_t) * m_numOfThread);
    ThreadInfo *pThreadInfo = (ThreadInfo *)malloc(sizeof(ThreadInfo) * m_numOfThread);

    for (int i = 0; i < m_numOfThread; i++) {
        pThreadInfo[i].m_NN        = (void *)this;
        pThreadInfo[i].m_threadNum = i;
        pthread_create(&(pThread[i]), NULL, ForwardPropagateForThread, (void *)&(pThreadInfo[i]));
    }

    for (int i = 0; i < m_numOfThread; i++) {
        pthread_join(pThread[i], NULL);
    }

    free(pThread);
    free(pThreadInfo);

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::TrainingOnGPU() {
#ifdef __CUDNN__
    this->ResetOperatorResult();
    this->ResetOperatorGradient();
    this->ResetLossFunctionResult();
    this->ResetLossFunctionGradient();

    this->ForwardPropagateOnGPU();
    this->BackPropagateOnGPU();

    m_aOptimizer->UpdateVariableOnGPU();
#else  // __CUDNN__
    std::cout << "There is no GPU option!" << '\n';
    exit(-1);
#endif  // __CUDNN__

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::TestingOnGPU() {
#ifdef __CUDNN__
    this->ResetOperatorResult();
    this->ResetLossFunctionResult();

    this->ForwardPropagateOnGPU();
#else  // __CUDNN__
    std::cout << "There is no GPU option!" << '\n';
    exit(-1);
#endif  // __CUDNN__

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

#ifdef __CUDNN__

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceGPU() {
    // std::cout << "NeuralNetwork<DTYPE>::SetModeGPU()" << '\n';
    m_Device = GPU;
    this->AllocOnGPU();

    for (int i = 0; i < m_OperatorDegree; i++) {
        // important order
        (*m_aaOperator)[i]->SetDeviceGPU(m_cudnnHandle);
    }
    m_aLossFunction->SetDeviceGPU(m_cudnnHandle);

    m_aOptimizer->SetCudnnHandle(m_cudnnHandle);
}

#endif  // __CUDNN__

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceCPU() {
    m_Device = CPU;

    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->SetDeviceCPU();
    }
    m_aLossFunction->SetDeviceCPU();
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceCPU(int pNumOfThread) {
    m_Device      = CPU;
    m_numOfThread = pNumOfThread;

    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->SetDeviceCPU(pNumOfThread);
    }
    m_aLossFunction->SetDeviceCPU(pNumOfThread);
}

// =========

template<typename DTYPE> int NeuralNetwork<DTYPE>::CreateGraph() {
    // in this part, we can check dependency between operator

    return TRUE;
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::PrintGraphInformation() {
    std::cout << "Graph Structure: " << "\n\n";

    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->PrintInformation();
        std::cout << '\n';
    }
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

template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetLossFunctionResult() {
    m_aLossFunction->ResetResult();
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetLossFunctionGradient() {
    m_aLossFunction->ResetGradient();
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

        if (name == pName) return (*m_aaOperator)[i];
    }

    return NULL;
}
