#include "Layer.h"

template class Layer<int>;
template class Layer<float>;
template class Layer<double>;

template<typename DTYPE> Layer<DTYPE>::Layer(std::string pName) {
    std::cout << "Layer<DTYPE>::Layer()" << '\n';
    m_aaOperator  = NULL;
    m_aaParameter = NULL;
    m_aaLayer     = NULL;

    m_numOfOperator  = 0;
    m_numOfParameter = 0;
    m_numOfLayer     = 0;
    m_name           = pName;
    Alloc();
}

template<typename DTYPE> Layer<DTYPE>::~Layer() {
    std::cout << "Layer<DTYPE>::~Layer()" << '\n';

    this->Delete();
}

template<typename DTYPE> int Layer<DTYPE>::Alloc() {
    m_aaOperator  = new Container<Operator<DTYPE> *>();
    m_aaParameter = new Container<Tensorholder<DTYPE> *>();
    m_aaLayer     = new Container<Layer<DTYPE> *>();
    return TRUE;
}

template<typename DTYPE> void Layer<DTYPE>::Delete() {
    std::cout << "Layer<DTYPE>::Delete()" << '\n';

    if (m_aaOperator) {
        Operator<DTYPE> **OperatorContainer = m_aaOperator->GetRawData();

        for (int i = 0; i < m_numOfOperator; i++) {
            delete OperatorContainer[i];
            OperatorContainer[i] = NULL;
        }
        delete m_aaOperator;
        m_aaOperator = NULL;
    }

    if (m_aaParameter) {
        Tensorholder<DTYPE> **ParameterContainer = m_aaParameter->GetRawData();

        for (int i = 0; i < m_numOfParameter; i++) {
            delete ParameterContainer[i];
            ParameterContainer[i] = NULL;
        }
        delete m_aaParameter;
        m_aaParameter = NULL;
    }

    if (m_aaLayer) {
        Layer<DTYPE> **LayerContainer = m_aaLayer->GetRawData();

        for (int i = 0; i < m_numOfLayer; i++) {
            delete LayerContainer[i];
            LayerContainer[i] = NULL;
        }
        delete m_aaLayer;
        m_aaLayer = NULL;
    }
}

template<typename DTYPE> Operator<DTYPE> *Layer<DTYPE>::AddLayer(Layer<DTYPE> *pLayer) {
    int pNumOfOperator  = pLayer->GetNumOfOperator();
    int pNumOfParameter = pLayer->GetNumOfParameter();

    for (int i = 0; i < pNumOfOperator; i++) {
        m_aaOperator->Push(pLayer->PopOperator());
        m_numOfOperator++;
    }

    for (int i = 0; i < pNumOfParameter; i++) {
        m_aaParameter->Push(pLayer->PopParameter());
        m_numOfParameter++;
    }

    m_aaLayer->Push(pLayer);
    m_numOfLayer++;

    return m_aaOperator->GetLast();
}

template<typename DTYPE> Operator<DTYPE> *Layer<DTYPE>::AddOperator(Operator<DTYPE> *pOperator) {
    m_aaOperator->Push(pOperator);
    m_numOfOperator++;

    return pOperator;
}

template<typename DTYPE> Tensorholder<DTYPE> *Layer<DTYPE>::AddParameter(Tensorholder<DTYPE> *pParameter) {
    m_aaParameter->Push(pParameter);
    m_numOfParameter++;

    return pParameter;
}

template<typename DTYPE> Container<Operator<DTYPE> *> *Layer<DTYPE>::GetOperatorContainer() {
    return m_aaOperator;
}

template<typename DTYPE> Container<Tensorholder<DTYPE> *> *Layer<DTYPE>::GetParameterContainer() {
    return m_aaParameter;
}

template<typename DTYPE> int Layer<DTYPE>::GetNumOfOperator() {
    return m_numOfOperator;
}

template<typename DTYPE> int Layer<DTYPE>::GetNumOfParameter() {
    return m_numOfParameter;
}

template<typename DTYPE> std::string Layer<DTYPE>::GetName() {
    return m_name;
}

template<typename DTYPE> Operator<DTYPE> *Layer<DTYPE>::PopOperator() {
    m_numOfOperator--;
    return m_aaOperator->Pop();
}

template<typename DTYPE> Tensorholder<DTYPE> *Layer<DTYPE>::PopParameter() {
    m_numOfParameter--;
    return m_aaParameter->Pop();
}
