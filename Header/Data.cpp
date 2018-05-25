#include "Data.h"

template class Data<int>;
template class Data<float>;
template class Data<double>;

template<typename DTYPE> int Data<DTYPE>::Alloc() {
    m_aaHostData = new DTYPE *[m_TimeSize];

    for (int i = 0; i < m_TimeSize; i++) {
        m_aaHostData[i] = new DTYPE[m_CapacityPerTime];

        for (int j = 0; j < m_CapacityPerTime; j++) {
            m_aaHostData[i][j] = 0.f;
        }
    }

    m_Device = CPU;

    return TRUE;
}

template<typename DTYPE> int Data<DTYPE>::Alloc(Data *pData) {
    m_aaHostData = new DTYPE *[m_TimeSize];

    for (int i = 0; i < m_TimeSize; i++) {
        m_aaHostData[i] = new DTYPE[m_CapacityPerTime];

        for (int j = 0; j < m_CapacityPerTime; j++) {
            m_aaHostData[i][j] = (*pData)[i * m_CapacityPerTime + j];
        }
    }

    m_Device = pData->GetDevice();

#ifdef __CUDNN__

    if (m_Device == GPU) pData->SetDeviceGPU();
#endif  // if __CUDNN__

    return TRUE;
}

template<typename DTYPE> void Data<DTYPE>::Delete() {
    if (m_aaHostData) {
        for (int i = 0; i < m_TimeSize; i++) {
            if (m_aaHostData[i]) {
                delete[] m_aaHostData[i];
                m_aaHostData[i] = NULL;
            }
        }
        delete[] m_aaHostData;
        m_aaHostData = NULL;
    }

#if __CUDNN__

    this->DeleteOnGPU();
#endif  // __CUDNN__
}

#ifdef __CUDNN__

template<typename DTYPE> int Data<DTYPE>::AllocOnGPU() {
    if (m_aaDevData == NULL) {
        m_aaDevData = new DTYPE *[m_TimeSize];

        for (int i = 0; i < m_TimeSize; i++) {
            checkCudaErrors(cudaMalloc((void **)&(m_aaDevData[i]), (m_CapacityPerTime * sizeof(DTYPE))));
        }
    }
    return TRUE;
}

template<typename DTYPE> void Data<DTYPE>::DeleteOnGPU() {
    if (m_aaDevData) {
        for (int i = 0; i < m_TimeSize; i++) {
            if (m_aaDevData[i]) {
                checkCudaErrors(cudaFree(m_aaDevData[i]));
                m_aaDevData[i] = NULL;
            }
        }
        delete[] m_aaDevData;
        m_aaDevData = NULL;
    }
}

template<typename DTYPE> int Data<DTYPE>::MemcpyCPU2GPU() {
    if (m_aaDevData != NULL) {
        for (int i = 0; i < m_TimeSize; i++) {
            checkCudaErrors(cudaMemcpy(m_aaDevData[i], m_aaHostData[i], (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyHostToDevice));
        }
    }
    return TRUE;
}

template<typename DTYPE> int Data<DTYPE>::MemcpyGPU2CPU() {
    if (m_aaDevData != NULL) {
        for (int i = 0; i < m_TimeSize; i++) {
            checkCudaErrors(cudaMemcpy(m_aaHostData[i], m_aaDevData[i], (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToHost));
        }
    }
    return TRUE;
}

#endif  // if __CUDNN__

template<typename DTYPE> Data<DTYPE>::Data(unsigned int pTimeSize, unsigned int pCapacity) {
    #if __DEBUG__
    std::cout << "Data<DTYPE>::Data(Shape *)" << '\n';
    #endif  // __DEBUG__
    m_TimeSize        = pTimeSize;
    m_CapacityPerTime = pCapacity;
    m_aaHostData      = NULL;
    m_Device          = CPU;
#if __CUDNN__
    m_aaDevData = NULL;
#endif  // __CUDNN
    Alloc();
}

template<typename DTYPE> Data<DTYPE>::Data(Data *pData) {
    #if __DEBUG__
    std::cout << "Data<DTYPE>::Data(Data *)" << '\n';
    #endif  // __DEBUG__
    m_TimeSize        = pData->GetTimeSize();
    m_CapacityPerTime = pData->GetCapacityPerTime();
    m_aaHostData      = NULL;
    m_Device          = CPU;
#if __CUDNN__
    m_aaDevData = NULL;
#endif  // __CUDNN
    Alloc(pData);
}

template<typename DTYPE> Data<DTYPE>::~Data() {
    #if __DEBUG__
    std::cout << "Data<DTYPE>::~Data()" << '\n';
    #endif  // __DEBUG__
    Delete();
}

template<typename DTYPE> int Data<DTYPE>::GetCapacity() {
    return m_TimeSize * m_CapacityPerTime;
}

template<typename DTYPE> int Data<DTYPE>::GetTimeSize() {
    return m_TimeSize;
}

template<typename DTYPE> int Data<DTYPE>::GetCapacityPerTime() {
    return m_CapacityPerTime;
}

template<typename DTYPE> DTYPE& Data<DTYPE>::operator[](unsigned int index) {
    return m_aaHostData[index / m_CapacityPerTime][index % m_CapacityPerTime];
}

template<typename DTYPE> Device Data<DTYPE>::GetDevice() {
    return m_Device;
}

template<typename DTYPE> DTYPE *Data<DTYPE>::GetCPUData(unsigned int pTime) {
    return m_aaHostData[pTime];
}

#ifdef __CUDNN__

template<typename DTYPE> DTYPE *Data<DTYPE>::GetGPUData(unsigned int pTime) {
    return m_aaDevData[pTime];
}

template<typename DTYPE> int Data<DTYPE>::SetDeviceCPU() {
    this->MemcpyGPU2CPU();
    return TRUE;
}

template<typename DTYPE> int Data<DTYPE>::SetDeviceGPU() {
    if (m_aaDevData == NULL) this->AllocOnGPU();
    this->MemcpyCPU2GPU();
    return TRUE;
}

#endif  // if __CUDNN__

//// example code
// int main(int argc, char const *argv[]) {
// Data<int> *pData = new Data<int>(2048);
//
// std::cout << pData->GetCapacity() << '\n';
// std::cout << (*pData)[2048] << '\n';
//
// delete pData;
//
// return 0;
// }
