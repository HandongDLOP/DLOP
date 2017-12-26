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

template<typename DTYPE> Operator<DTYPE>::Operator(Tensor<DTYPE> *pTensor, std::string pName) {
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
    Alloc(pTensor);
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

template<typename DTYPE> int Operator<DTYPE>::Alloc(Tensor<DTYPE> *pTensor) {
    std::cout << "Operator<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';

    m_aResult = pTensor;

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::Alloc(int numInput, ...) {
    std::cout << "Operator<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';

    va_list ap;
    va_start(ap, numInput);

    for (int i = 0; i < numInput; i++) {
        this->AddEdgebetweenOperators(va_arg(ap, Operator<DTYPE> *));
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

// Add Graph Edge
template<typename DTYPE> int Operator<DTYPE>::_AddInputEdge(Operator<DTYPE> *pInput) {
    Operator<DTYPE> **temp = new Operator<DTYPE> *[m_InputDegree + 1];

    for (int i = 0; i < m_InputDegree; i++) temp[i] = m_apInput[i];
    temp[m_InputDegree] = pInput;

    if (m_apInput) {
        delete[] m_apInput;
        m_apInput = NULL;
    }

    m_apInput = temp;

    m_InputDegree++;

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::_AddOutputEdge(Operator<DTYPE> *pOutput) {
    Operator<DTYPE> **temp = new Operator<DTYPE> *[m_OutputDegree + 1];

    for (int i = 0; i < m_OutputDegree; i++) temp[i] = m_apOutput[i];
    temp[m_OutputDegree] = pOutput;

    if (m_apOutput) {
        delete[] m_apOutput;
        m_apOutput = NULL;
    }

    m_apOutput = temp;

    m_OutputDegree++;

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::AddEdgebetweenOperators(Operator<DTYPE> *pInput) {
    this->_AddInputEdge(pInput);
    pInput->_AddOutputEdge(this);

    return TRUE;
}

template<typename DTYPE> void Operator<DTYPE>::IncreaseCurrentOutputDegree() {
    m_currentOutputDegree++;
}

template<typename DTYPE> void Operator<DTYPE>::IncreaseCurrentInputDegree() {
    m_currentInputDegree++;
}

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

// ===========================================================================================

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
    return TRUE;
}

int main(int argc, char const *argv[]) {
    Operator<int> *temp1 = new Operator<int>(new Tensor<int>(1, 100, 1, 28, 28), "temp1");
    Operator<int> *temp2 = new Operator<int>(temp1, "temp2");
    Operator<int> *temp3 = new Operator<int>(temp1, temp2, "temp3");

    std::cout << temp3->GetInput()[0]->GetName() << '\n';
    std::cout << temp3->GetInput()[1]->GetName() << '\n';

    delete temp1;
    delete temp2;
    delete temp3;

    return 0;
}
