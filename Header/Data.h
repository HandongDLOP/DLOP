#ifndef __DATA__
#define __DATA__    value

#include "Common.h"

template<typename DTYPE> class Data {
private:
    DTYPE **m_aaHostData;

    int m_TimeSize;
    int m_CapacityPerTime;

    Device m_Device;

#if __CUDNN__
    DTYPE **m_aaDevData;
#endif  // __CUDNN

private:
    int  Alloc();
    int  Alloc(Data *pData);
    void Delete();

#if __CUDNN__
    int  AllocOnGPU();
    void DeleteOnGPU();
    int  MemcpyCPU2GPU();
    int  MemcpyGPU2CPU();
#endif  // __CUDNN

public:
    Data(unsigned int pCapacity);
    Data(unsigned int pTimeSize, unsigned int pCapacityPerTime);
    Data(Data *pData);  // Copy Constructor
    virtual ~Data();

    int    GetCapacity();
    int    GetTimeSize();
    int    GetCapacityPerTime();
    DTYPE  GetElement(unsigned int index);
    DTYPE& operator[](unsigned int index);
    Device GetDevice();
    DTYPE* GetCPUData(unsigned int pTime = 0);

#ifdef __CUDNN__
    int    SetDeviceCPU();
    int    SetDeviceGPU();

    DTYPE* GetGPUData(unsigned int pTime = 0);

#endif  // if __CUDNN__
};


#endif  // __DATA__
