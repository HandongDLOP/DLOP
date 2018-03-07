#include "Operator.h"

template class Operator<int>;
template class Operator<float>;
template class Operator<double>;

template<typename DTYPE> Operator<DTYPE>::Operator(std::string pName) {
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    m_aResult             = NULL;
    m_aGradient           = NULL;
    m_aDelta              = NULL;
    m_apOutput            = NULL;
    m_apInput             = NULL;
    m_OutputDegree        = 0;
    m_InputDegree         = 0;
    m_currentOutputDegree = 0;
    m_currentInputDegree  = 0;
    m_name                = pName;
}

template<typename DTYPE> Operator<DTYPE>::Operator(Operator<DTYPE> *pInput, std::string pName) {
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    m_aResult             = NULL;
    m_aGradient           = NULL;
    m_aDelta              = NULL;
    m_apOutput            = NULL;
    m_apInput             = NULL;
    m_OutputDegree        = 0;
    m_InputDegree         = 0;
    m_currentOutputDegree = 0;
    m_currentInputDegree  = 0;
    m_name                = pName;
    Alloc(1, pInput);
}

template<typename DTYPE> Operator<DTYPE>::Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) {
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    m_aResult             = NULL;
    m_aGradient           = NULL;
    m_aDelta              = NULL;
    m_apOutput            = NULL;
    m_apInput             = NULL;
    m_OutputDegree        = 0;
    m_InputDegree         = 0;
    m_currentOutputDegree = 0;
    m_currentInputDegree  = 0;
    m_name                = pName;
    Alloc(2, pInput0, pInput1);
}

template<typename DTYPE> Operator<DTYPE>::~Operator() {
    std::cout << "Operator<DTYPE>::~Operator()" << '\n';
    this->Delete();
}

template<typename DTYPE> int Operator<DTYPE>::Alloc(int numInput, ...) {
    std::cout << "Operator<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
    Operator<DTYPE> *temp = NULL;

    va_list ap;
    va_start(ap, numInput);

    for (int i = 0; i < numInput; i++) {
        temp = va_arg(ap, Operator<DTYPE> *);

        if (temp) {
            this->AddEdgebetweenOperators(temp);
        } else {
            for (int j = i - 1; j > -1; j--) {
                delete m_apInput[j];
                m_apInput[j] = NULL;
            }
            delete[] m_apInput;
            m_apInput = NULL;

            printf("Receive NULL pointer of Operator<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }
    }

    va_end(ap);

    return TRUE;
}

template<typename DTYPE> void Operator<DTYPE>::Delete() {
    if (m_aResult) {
        delete m_aResult;
        m_aResult = NULL;
    }

    if (m_aGradient) {
        delete m_aGradient;
        m_aGradient = NULL;
    }

    if (m_aDelta) {
        delete m_aDelta;
        m_aDelta = NULL;
    }

    if (m_apOutput) {
        delete[] m_apOutput;
        m_apOutput = NULL;
    }

    if (m_apInput) {
        delete[] m_apInput;
        m_apInput = NULL;
    }
}

#if __CUDNN__
template<typename DTYPE> void Operator<DTYPE>::SetCudnnHandle(cudnnHandle_t& pCudnnHandle) {
    m_pCudnnHandle = pCudnnHandle;
}

void cudnnResize(int size, float *data) {
    if (data == NULL) {
        checkCudaErrors(cudaFree(data));
    }
    checkCudaErrors(cudaMalloc(&data, size * sizeof(float)));
}

#endif // if __CUDNN__

template<typename DTYPE> void Operator<DTYPE>::SetResult(Tensor<DTYPE> *pTensor) {
    if (m_aResult) {
        delete m_aResult;
        m_aResult = NULL;
    }

    m_aResult = pTensor;
}

template<typename DTYPE> void Operator<DTYPE>::SetGradient(Tensor<DTYPE> *pTensor) {
    if (m_aGradient) {
        delete m_aGradient;
        m_aGradient = NULL;
    }

    m_aGradient = pTensor;
}

template<typename DTYPE> void Operator<DTYPE>::SetDelta(Tensor<DTYPE> *pTensor) {
    if (m_aDelta) {
        delete m_aDelta;
        m_aDelta = NULL;
    }

    m_aDelta = pTensor;
}

template<typename DTYPE> void Operator<DTYPE>::IncreaseCurrentOutputDegree() {
    m_currentOutputDegree++;
}

template<typename DTYPE> void Operator<DTYPE>::IncreaseCurrentInputDegree() {
    m_currentInputDegree++;
}

#if __CUDNN__
template<typename DTYPE> cudnnHandle_t& Operator<DTYPE>::GetCudnnHandle() {
    // return this->m_pCudnnHandle;
    return m_pCudnnHandle;
}

#endif // if __CUDNN__

template<typename DTYPE> Tensor<DTYPE> *Operator<DTYPE>::GetResult() const {
    return m_aResult;
}

template<typename DTYPE> Tensor<DTYPE> *Operator<DTYPE>::GetGradient() const {
    return m_aGradient;
}

template<typename DTYPE> Tensor<DTYPE> *Operator<DTYPE>::GetDelta() const {
    return m_aDelta;
}

template<typename DTYPE> Operator<DTYPE> **Operator<DTYPE>::GetOutput() const {
    return m_apOutput;
}

template<typename DTYPE> Operator<DTYPE> **Operator<DTYPE>::GetInput() const {
    return m_apInput;
}

template<typename DTYPE> int Operator<DTYPE>::GetOutputDegree() const {
    return m_OutputDegree;
}

template<typename DTYPE> int Operator<DTYPE>::GetInputDegree() const {
    return m_InputDegree;
}

template<typename DTYPE> int Operator<DTYPE>::GetCurrentOutputDegree() const {
    return m_currentOutputDegree;
}

template<typename DTYPE> int Operator<DTYPE>::GetCurrentInputDegree() const {
    return m_currentInputDegree;
}

template<typename DTYPE> std::string Operator<DTYPE>::GetName() const {
    return m_name;
}

// Add Graph Edge
template<typename DTYPE> int Operator<DTYPE>::_AddInputEdge(Operator<DTYPE> *pInput) {
    try {
        Operator<DTYPE> **temp = new Operator<DTYPE> *[m_InputDegree + 1];

        for (int i = 0; i < m_InputDegree; i++) temp[i] = m_apInput[i];
        temp[m_InputDegree] = pInput;

        if (m_apInput) {
            delete[] m_apInput;
            m_apInput = NULL;
        }

        m_apInput = temp;
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }
    m_InputDegree++;

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::_AddOutputEdge(Operator<DTYPE> *pOutput) {
    try {
        Operator<DTYPE> **temp = new Operator<DTYPE> *[m_OutputDegree + 1];

        for (int i = 0; i < m_OutputDegree; i++) temp[i] = m_apOutput[i];
        temp[m_OutputDegree] = pOutput;

        if (m_apOutput) {
            delete[] m_apOutput;
            m_apOutput = NULL;
        }

        m_apOutput = temp;
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }
    m_OutputDegree++;

    return TRUE;
}

template<typename DTYPE> void Operator<DTYPE>::AddEdgebetweenOperators(Operator<DTYPE> *pInput) {
    this->_AddInputEdge(pInput);
    pInput->_AddOutputEdge(this);
}

template<typename DTYPE> int Operator<DTYPE>::ForwardPropagate() {
    if (m_InputDegree == m_currentInputDegree) {
        this->ComputeForwardPropagate();

        for (int o = 0; o < m_OutputDegree; o++) {
            m_apOutput[o]->IncreaseCurrentInputDegree();
        }
        m_currentInputDegree = 0;
    } else {
        for (int i = 0; i < m_InputDegree; i++) {
            m_apInput[i]->ForwardPropagate();

            if (m_InputDegree == m_currentInputDegree) {
                this->ComputeForwardPropagate();

                for (int o = 0; o < m_OutputDegree; o++) {
                    m_apOutput[o]->IncreaseCurrentInputDegree();
                }
                m_currentInputDegree = 0;
            }
        }
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::ComputeForwardPropagate() {
    std::cout << this->GetName() << '\n';
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::BackPropagate() {
    this->ComputeBackPropagate();

    // value 조정
    for (int i = 0; i < m_InputDegree; i++) {
        if (m_apInput[i] != NULL) m_apInput[i]->IncreaseCurrentOutputDegree();
    }
    m_currentOutputDegree = 0;

    for (int i = 0; i < m_InputDegree; i++) {
        if (m_apInput[i]->GetOutputDegree() == m_apInput[i]->GetCurrentOutputDegree()) {
            m_apInput[i]->BackPropagate();
        }
    }
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::ComputeBackPropagate() {
    std::cout << this->GetName() << '\n';
    return TRUE;
}

template<typename DTYPE> Operator<DTYPE> *Operator<DTYPE>::Concatenate(Operator<DTYPE> *src, Operator<DTYPE> *dst, int axis) {
    switch (axis) {
        case 0: return Concatenate0(src, dst); // break;
        case 1: return Concatenate1(src, dst); // break;
        case 2: return Concatenate2(src, dst); // break;
        default:
            printf("Receive invalid axis value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
    }
}

template<typename DTYPE> int IsSameDim(Operator<DTYPE> *src, Operator<DTYPE> *dst) {
    if ((src->GetResult()->GetTimeSize() != dst->GetResult()->GetTimeSize()) // src->GetResult()->GetChannelSize() != dst->GetResult()->GetChannelSize() ||
        || (src->GetResult()->GetRowSize() != dst->GetResult()->GetRowSize()) || (src->GetResult()->GetColSize() != dst->GetResult()->GetColSize())) {
        printf("Receive invalid tensor in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    return TRUE;
}

// \sigma(n_i) * ch_1 * h * w
template<typename DTYPE> Operator<DTYPE>* Concatenate0(Operator<DTYPE> *src, Operator<DTYPE> *dst) {
    if (IsSameDim(src, dst)) {
        int batch = src->GetResult()->GetBatchSize();
    }
}

// n_1 * \sigma(ch_i) * h * w
template<typename DTYPE> Operator<DTYPE>* Concatenate1(Operator<DTYPE> *src, Operator<DTYPE> *dst) {
    if (IsSameDim(src, dst)) {}
}

//
template<typename DTYPE> Operator<DTYPE>* Concatenate2(Operator<DTYPE> *src1, Operator<DTYPE> *src2) {
    if (IsSameDim(src1, src2)) {
        Operator<DTYPE> *concat = new Operator<DTYPE>("concat");

        Tensor<DTYPE> *srcTensor1 = src1->GetResult();
        Tensor<DTYPE> *srcTensor2 = src2->GetResult();
        Tensor<DTYPE> *srcDelta1  = src1->GetDelta();
        Tensor<DTYPE> *srcDelta2  = src2->GetDelta();

        int t          = srcTensor1->GetTimeSize();
        int b          = srcTensor1->GetBatchSize();
        int newChannel = srcTensor1->GetChannelSize() + srcTensor2->GetChannelSize();
        int r          = srcTensor1->GetRowSize();
        int c          = srcTensor1->GetColSize();

        Tensor<DTYPE> *newTensor = new Tensor<DTYPE>(t, b, newChannel, r, c);
        Tensor<DTYPE> *newDelta  = new Tensor<DTYPE>(t, b, newChannel, r, c);

        int i            = 0;
        int srcCapacity1 = src1->GetResult()->GetData()->GetCapacity();
        int newCapacity  = newTensor->GetData()->GetCapacity();

        for (i = 0; i < srcCapacity1; i++) {
            (*newTensor)[i] = (*srcTensor1)[i];
            (*newDelta)[i]  = (*srcDelta1)[i];
        }

        for (i = srcCapacity1; i < newCapacity; i++) {
            (*newTensor)[i] = (*srcTensor2)[i - srcCapacity1];
            (*newDelta)[i]  = (*srcDelta2)[i - srcCapacity1];
        }

        concat->SetResult(newTensor);
        concat->SetDelta(newDelta);

        return concat;
    }
}

// int main(int argc, char const *argv[]) {
// Operator<int> *temp1 = new Operator<int>("temp1");
// Operator<int> *temp2 = new Operator<int>(temp1, "temp2");
// Operator<int> *temp3 = new Operator<int>(temp1, temp2, "temp3");
//
// std::cout << temp3->GetInput()[0]->GetName() << '\n';
// std::cout << temp3->GetInput()[1]->GetName() << '\n';
// std::cout << temp1->GetOutput()[0]->GetName() << '\n';
//
// delete temp1;
// delete temp2;
// delete temp3;
//
// return 0;
// }
