#include "Tensor.h"

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;

//////////////////////////////////////////////////////////////////////////////// for private method

template<typename DTYPE> int Tensor<DTYPE>::Alloc(Shape *pShape, IsUseTime pAnswer) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Alloc(Shape *pShape, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    if (pShape == NULL) {
        printf("Receive NULL pointer of Shape class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape    = pShape;
        m_IsUseTime = pAnswer;

        int rank = pShape->GetRank();

        int pTime            = 1;
        int pCapacityPerTime = 1;

        if (m_IsUseTime == UseTime) {
            pTime = (*pShape)[0];

            for (int i = 1; i < rank; i++) {
                pCapacityPerTime *= (*pShape)[i];
            }
        } else if (m_IsUseTime == NoUseTime) {
            for (int i = 0; i < rank; i++) {
                pCapacityPerTime *= (*pShape)[i];
            }
        } else return FALSE;

        m_aData = new Data<DTYPE>(pTime, pCapacityPerTime);
    }

    m_Device = CPU;

    return TRUE;
}

template<typename DTYPE> int Tensor<DTYPE>::Alloc(Tensor<DTYPE> *pTensor) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Alloc(Tensor<DTYPE> *pTensor)" << '\n';
    #endif  // __DEBUG__

    if (pTensor == NULL) {
        printf("Receive NULL pointer of Tensor<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape    = new Shape(pTensor->GetShape());
        m_aData     = new Data<DTYPE>(pTensor->GetData());
        m_Device    = pTensor->GetDevice();
        m_IsUseTime = pTensor->GetIsUseTime();
    }

    return TRUE;
}

template<typename DTYPE> void Tensor<DTYPE>::Delete() {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Delete()" << '\n';
    #endif  // __DEBUG__

    if (m_aShape) {
        delete m_aShape;
        m_aShape = NULL;
    }

    if (m_aData) {
        delete m_aData;
        m_aData = NULL;
    }
}

//////////////////////////////////////////////////////////////////////////////// for public method

template<typename DTYPE> Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape = NULL;
    m_aData  = NULL;
    m_Device = CPU;
    Alloc(new Shape(pSize0, pSize1, pSize2, pSize3, pSize4), pAnswer);
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, int pSize3, IsUseTime pAnswer) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, int pSize3, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape = NULL;
    m_aData  = NULL;
    m_Device = CPU;
    Alloc(new Shape(pSize0, pSize1, pSize2, pSize3), pAnswer);
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, IsUseTime pAnswer) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(int pSize0, int pSize1, int pSize2, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape = NULL;
    m_aData  = NULL;
    m_Device = CPU;
    Alloc(new Shape(pSize0, pSize1, pSize2), pAnswer);
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(int pSize0, int pSize1, IsUseTime pAnswer) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(int pSize0, int pSize1, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape = NULL;
    m_aData  = NULL;
    m_Device = CPU;
    Alloc(new Shape(pSize0, pSize1), pAnswer);
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(int pSize0, IsUseTime pAnswer) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(int pSize0, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape = NULL;
    m_aData  = NULL;
    m_Device = CPU;
    Alloc(new Shape(pSize0), pAnswer);
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(Shape *pShape, IsUseTime pAnswer) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(Shape *pShape, IsUseTime pAnswer)" << '\n';
    #endif  // __DEBUG__

    m_aShape = NULL;
    m_aData  = NULL;
    m_Device = CPU;
    Alloc(pShape, pAnswer);
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(Tensor *pTensor) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Tensor(Tensor *pTensor)" << '\n';
    #endif  // __DEBUG__

    m_aShape = NULL;
    m_aData  = NULL;
    m_Device = CPU;
    Alloc(pTensor);
}

template<typename DTYPE> Tensor<DTYPE>::~Tensor() {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::~Tensor()" << '\n';
    #endif  // __DEBUG__

    Delete();
}

template<typename DTYPE> Shape *Tensor<DTYPE>::GetShape() {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetShape()" << '\n';
    #endif  // __DEBUG__

    return m_aShape;
}

template<typename DTYPE> int Tensor<DTYPE>::GetRank() {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetRank()" << '\n';
    #endif  // __DEBUG__

    return m_aShape->GetRank();
}

template<typename DTYPE> int Tensor<DTYPE>::GetDim(int pRanknum) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetDim(int pRanknum)" << '\n';
    #endif  // __DEBUG__

    return m_aShape->GetDim(pRanknum);
}

template<typename DTYPE> Data<DTYPE> *Tensor<DTYPE>::GetData() {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetData()" << '\n';
    #endif  // __DEBUG__

    return m_aData;
}

template<typename DTYPE> int Tensor<DTYPE>::GetCapacity() {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetCapacity()" << '\n';
    #endif  // __DEBUG__

    return m_aData->GetCapacity();
}

template<typename DTYPE> int Tensor<DTYPE>::GetElement(unsigned int index) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetElement(unsigned int index)" << '\n';
    #endif  // __DEBUG__

    return m_aData->GetElement(index);
}

template<typename DTYPE> DTYPE& Tensor<DTYPE>::operator[](unsigned int index) {
    #if __CUDNN__
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::operator[](unsigned int index)" << '\n';

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    return (*m_aData)[index];
}

template<typename DTYPE> Device Tensor<DTYPE>::GetDevice() {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetDevice()" << '\n';
    #endif  // __DEBUG__

    return m_Device;
}

template<typename DTYPE> IsUseTime Tensor<DTYPE>::GetIsUseTime() {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetIsUseTime()" << '\n';
    #endif  // __DEBUG__

    return m_IsUseTime;
}

template<typename DTYPE> DTYPE *Tensor<DTYPE>::GetCPUData(unsigned int pTime) {
    #if __CUDNN__
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetCPUData(unsigned int pTime)" << '\n';

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    return m_aData->GetCPUData(pTime);
}

template<typename DTYPE> int Tensor<DTYPE>::GetTimeSize() {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetTimeSize()" << '\n';
    #endif  // __DEBUG__

    if ((m_aShape->GetRank() == 5) && (m_IsUseTime == UseTime)) return (*m_aShape)[0];
    else return 0;
}

template<typename DTYPE> int Tensor<DTYPE>::GetBatchSize() {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetBatchSize()" << '\n';
    #endif  // __DEBUG__

    if ((m_aShape->GetRank() == 5) && (m_IsUseTime == UseTime)) return (*m_aShape)[1];
    else return 0;
}

template<typename DTYPE> int Tensor<DTYPE>::GetChannelSize() {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetChannelSize()" << '\n';
    #endif  // __DEBUG__

    if ((m_aShape->GetRank() == 5) && (m_IsUseTime == UseTime)) return (*m_aShape)[2];
    else return 0;
}

template<typename DTYPE> int Tensor<DTYPE>::GetRowSize() {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetRowSize()" << '\n';
    #endif  // __DEBUG__

    if ((m_aShape->GetRank() == 5) && (m_IsUseTime == UseTime)) return (*m_aShape)[3];
    else return 0;
}

template<typename DTYPE> int Tensor<DTYPE>::GetColSize() {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetColSize()" << '\n';
    #endif  // __DEBUG__

    if ((m_aShape->GetRank() == 5) && (m_IsUseTime == UseTime)) return (*m_aShape)[4];
    else return 0;
}

template<typename DTYPE> int Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4)" << '\n';
    #endif  // __DEBUG__

    int cur_capacity = GetCapacity();
    int new_capacity = pSize0 * pSize1 * pSize2 * pSize3 * pSize4;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot ReShape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(5, pSize0, pSize1, pSize2, pSize3, pSize4);
    }

    return TRUE;
}

template<typename DTYPE> int Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2, int pSize3) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2, int pSize3)" << '\n';
    #endif  // __DEBUG__

    int cur_capacity = GetCapacity();
    int new_capacity = pSize0 * pSize1 * pSize2 * pSize3;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot ReShape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(4, pSize0, pSize1, pSize2, pSize3);
    }

    return TRUE;
}

template<typename DTYPE> int Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::ReShape(int pSize0, int pSize1, int pSize2)" << '\n';
    #endif  // __DEBUG__

    int cur_capacity = GetCapacity();
    int new_capacity = pSize0 * pSize1 * pSize2;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot ReShape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(3, pSize0, pSize1, pSize2);
    }

    return TRUE;
}

template<typename DTYPE> int Tensor<DTYPE>::ReShape(int pSize0, int pSize1) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::ReShape(int pSize0, int pSize1)" << '\n';
    #endif  // __DEBUG__

    int cur_capacity = GetCapacity();
    int new_capacity = pSize0 * pSize1;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot ReShape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(2, pSize0, pSize1);
    }

    return TRUE;
}

template<typename DTYPE> int Tensor<DTYPE>::ReShape(int pSize0) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::ReShape(int pSize0)" << '\n';
    #endif  // __DEBUG__

    int cur_capacity = GetCapacity();
    int new_capacity = pSize0;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot ReShape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(1, pSize0);
    }

    return TRUE;
}

template<typename DTYPE> void Tensor<DTYPE>::Reset() {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Reset()" << '\n';
    #endif  // __DEBUG__

    int capacity = GetCapacity();

    #if __CUDNN__
    # if __DEBUG__

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->SetDeviceCPU();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->SetDeviceCPU();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    for (int i = 0; i < capacity; i++) {
        (*m_aData)[i] = 0;
    }
}

template<typename DTYPE> void Tensor<DTYPE>::SetDeviceCPU() {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::SetDeviceCPU()" << '\n';
    #endif  // __DEBUG__

    m_Device = CPU;
    m_aData->SetDeviceCPU();
    m_aShape->SetDeviceCPU();
}

#ifdef __CUDNN__
template<typename DTYPE> void Tensor<DTYPE>::SetDeviceGPU() {
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::SetDeviceGPU()" << '\n';
    # endif // __DEBUG__

    m_Device = GPU;
    m_aData->SetDeviceGPU();
    m_aShape->SetDeviceGPU();
}

template<typename DTYPE> DTYPE *Tensor<DTYPE>::GetGPUData(unsigned int pTime) {
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetGPUData(unsigned int pTime)" << '\n';

    if (m_Device == CPU) {
        printf("Warning! Tensor is allocated in Host(CPU) latest time\n");
        printf("Change mode CPU toGPU\n");
        this->SetDeviceGPU();
    }

    # else // if __DEBUG__

    if (m_Device == CPU) {
        this->SetDeviceGPU();
    }

    # endif // __DEBUG__

    return m_aData->GetGPUData(pTime);
}

template<typename DTYPE> cudnnTensorDescriptor_t& Tensor<DTYPE>::GetDescriptor() {
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::GetDescriptor()" << '\n';
    # endif // __DEBUG__

    return m_aShape->GetDescriptor();
}

template<typename DTYPE> void Tensor<DTYPE>::Reset(cudnnHandle_t& pCudnnHandle) {
    # if __DEBUG__
    std::cout << "Tensor<DTYPE>::Reset(cudnnHandle_t& pCudnnHandle)" << '\n';
    # endif // __DEBUG__

    int pTime                     = this->GetTimeSize();
    cudnnTensorDescriptor_t pDesc = this->GetDescriptor();
    DTYPE *pDevData               = NULL;
    float  zero                   = 0.f;

    for (int i = 0; i < pTime; i++) {
        pDevData = this->GetGPUData(i);
        checkCUDNN(cudnnAddTensor(pCudnnHandle,
                                  &zero, pDesc, pDevData,
                                  &zero, pDesc, pDevData));
    }
}

#endif  // if __CUDNN__

////////////////////////////////////////////////////////////////////////////////static method

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Truncated_normal(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, float mean, float stddev, IsUseTime pAnswer) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Truncated_normal()" << '\n';
    #endif  // __DEBUG__

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> rand(mean, stddev);

    Tensor<DTYPE> *temp = new Tensor<DTYPE>(pSize0, pSize1, pSize2, pSize3, pSize4, pAnswer);

    int capacity = temp->GetCapacity();

    for (int i = 0; i < capacity; i++) {
        (*temp)[i] = rand(gen);
    }

    return temp;
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Truncated_normal(Shape *pShape, float mean, float stddev, IsUseTime pAnswer) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Truncated_normal()" << '\n';
    #endif  // __DEBUG__

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> rand(mean, stddev);

    Tensor<DTYPE> *temp = new Tensor<DTYPE>(pShape, pAnswer);

    int capacity = temp->GetCapacity();

    for (int i = 0; i < capacity; i++) {
        (*temp)[i] = rand(gen);
    }

    return temp;
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Zeros(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Zero()" << '\n';
    #endif  // __DEBUG__

    return new Tensor<DTYPE>(pSize0, pSize1, pSize2, pSize3, pSize4, pAnswer);
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Zeros(Shape *pShape, IsUseTime pAnswer) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Zero()" << '\n';
    #endif  // __DEBUG__

    return new Tensor<DTYPE>(pShape, pAnswer);
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Constants(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, DTYPE constant, IsUseTime pAnswer) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Constant()" << '\n';
    #endif  // __DEBUG__

    Tensor<DTYPE> *temp = new Tensor<DTYPE>(pSize0, pSize1, pSize2, pSize3, pSize4, pAnswer);

    int capacity = temp->GetCapacity();

    for (int i = 0; i < capacity; i++) {
        (*temp)[i] = constant;
    }

    return temp;
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Constants(Shape *pShape, DTYPE constant, IsUseTime pAnswer) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Constant()" << '\n';
    #endif  // __DEBUG__

    Tensor<DTYPE> *temp = new Tensor<DTYPE>(pShape, pAnswer);

    int capacity = temp->GetCapacity();

    for (int i = 0; i < capacity; i++) {
        (*temp)[i] = constant;
    }

    return temp;
}

// example code
// int main(int argc, char const *argv[]) {
// Tensor<float> *left  = Tensor<float>::Constants(1, 2, 3, 3, 3, 2);
// Tensor<float> *right = Tensor<float>::Truncated_normal(1, 1, 3, 1, 1, 0.0, 0.1);
// Tensor<float> *dst   = Tensor<float>::Zeros(1, 2, 3, 3, 3);
//
// std::cout << left << '\n';
// std::cout << right << '\n';
// std::cout << dst << '\n';
//
// Tensor<float>::BroadcastAdd(left, right);
//
// std::cout << dst << '\n';
//
// return 0;
// }
