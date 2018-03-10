#include "Container.h"

template<typename DTYPE> class Tensor;
template<typename DTYPE> class Operator;

template class Container<Tensor<int> *>;
template class Container<Tensor<float> *>;
template class Container<Tensor<double> *>;

template class Container<Operator<int> *>;
template class Container<Operator<float> *>;
template class Container<Operator<double> *>;

template<typename DTYPE> Container<DTYPE>::Container(){
    m_aElement = NULL;
    m_size = 0;
}

template<typename DTYPE> Container<DTYPE>::~Container(){
    if(m_aElement){
        delete [] m_aElement;
        m_aElement = NULL;
    }
}

template<typename DTYPE> int Container<DTYPE>::Append(DTYPE pElement){
    try {
        DTYPE *temp = new DTYPE[m_size + 1];

        for (int i = 0; i < m_size; i++) temp[i] = m_aElement[i];
        temp[m_size] = pElement;

        if (m_aElement) {
            delete[] m_aElement;
            m_aElement = NULL;
        }

        m_aElement = temp;
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    m_size++;

    return TRUE;
}

template<typename DTYPE> DTYPE Container<DTYPE>::operator[](unsigned int index){
    return m_aElement[index];
}

// int main(int argc, char const *argv[]) {
//
// Container<Tensor<float> *> *p = new Container<Tensor<float> *>();
// Container<int> *_p = new Container<int>();
//
// return 0;
// }
