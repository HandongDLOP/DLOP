#include "Objective.h"

template class Objective<int>;
template class Objective<float>;
template class Objective<double>;

template<typename DTYPE> Objective<DTYPE>::Objective(std::string pName) {
    std::cout << "Objective<DTYPE>::Objective()" << '\n';
    m_aResult = NULL;
    m_apInput = NULL;
    m_InputDegree = 0;
    m_currentInputDegree = 0;
    m_name = pName;
}

template<typename DTYPE> Objective<DTYPE>::Objective(Operator<DTYPE> *pInput, std::string pName) {
    std::cout << "Objective<DTYPE>::Objective()" << '\n';
    m_aResult = NULL;
    m_apInput = NULL;
    m_InputDegree = 0;
    m_currentInputDegree = 0;
    m_name = pName;
    Alloc(1, pInput);
}

template<typename DTYPE> Objective<DTYPE>::Objective(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) {
    std::cout << "Objective<DTYPE>::Objective()" << '\n';
    m_aResult = NULL;
    m_apInput = NULL;
    m_InputDegree = 0;
    m_currentInputDegree = 0;
    m_name = pName;
    Alloc(2, pInput0, pInput1);
}

template<typename DTYPE> Objective<DTYPE>::~Objective() {
    std::cout << "Objective<DTYPE>::~Objective()" << '\n';
    this->Delete();
}

template<typename DTYPE> int Objective<DTYPE>::Alloc(int numInput, ...) {
    std::cout << "Objective<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
    Operator<DTYPE> *temp = NULL;

    va_list ap;
    va_start(ap, numInput);

    for (int i = 0; i < numInput; i++) {
        temp = va_arg(ap, Operator<DTYPE> *);

        if (temp) {
            this->_AddInputEdge(temp);
        } else {
            for (int j = i - 1; j > -1; j--) {
                delete m_apInput[j];
                m_apInput[j] = NULL;
            }
            delete[] m_apInput;
            m_apInput = NULL;

            printf("Receive NULL pointer of Objective<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }
    }

    va_end(ap);

    return TRUE;
}

template<typename DTYPE> void Objective<DTYPE>::Delete() {
    if (m_aResult) {
        delete m_aResult;
        m_aResult = NULL;
    }

    if (m_apInput) {
        delete[] m_apInput;
        m_apInput = NULL;
    }
}

template<typename DTYPE> void Objective<DTYPE>::SetResult(Tensor<DTYPE> *pTensor) {
    m_aResult = pTensor;
}

template<typename DTYPE> void Objective<DTYPE>::IncreaseCurrentInputDegree() {
    m_currentInputDegree++;
}

template<typename DTYPE> Tensor<DTYPE> *Objective<DTYPE>::GetResult() const {
    return m_aResult;
}

template<typename DTYPE> Operator<DTYPE> **Objective<DTYPE>::GetInput() const {
    return m_apInput;
}

template<typename DTYPE> int Objective<DTYPE>::GetInputDegree() const {
    return m_InputDegree;
}

template<typename DTYPE> int Objective<DTYPE>::GetCurrentInputDegree() const {
    return m_currentInputDegree;
}

template<typename DTYPE> std::string Objective<DTYPE>::GetName() const {
    return m_name;
}

// Add Graph Edge
template<typename DTYPE> int Objective<DTYPE>::_AddInputEdge(Operator<DTYPE> *pInput) {
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

template<typename DTYPE> int Objective<DTYPE>::ComputeForwardPropagate() {
    std::cout << this->GetName() << '\n';
    return TRUE;
}

template<typename DTYPE> int Objective<DTYPE>::ComputeBackPropagate() {
    std::cout << this->GetName() << '\n';
    return TRUE;
}

// int main(int argc, char const *argv[]) {
// Objective<int> *temp1 = new Objective<int>("temp1");
// Objective<int> *temp2 = new Objective<int>(temp1, "temp2");
// Objective<int> *temp3 = new Objective<int>(temp1, temp2, "temp3");
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
