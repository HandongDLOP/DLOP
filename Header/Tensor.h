#ifndef TENSOR_H_
#define TENSOR_H_

#include <time.h>
#include <math.h>
#include <chrono>
#include <random>

#include "Shape.h"
#include "Data.h"

enum IsUseTime {
    UseTime,
    NoUseTime
};

template<typename DTYPE> class Tensor {
private:
    Shape *m_aShape;
    Data<DTYPE> *m_aData;
    Device m_Device;
    IsUseTime m_IsUseTime;

private:
    int  Alloc(Shape *pShape, IsUseTime pAnswer);
    int  Alloc(Tensor *pTensor);
    void Delete();

public:
    Tensor(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4, IsUseTime pAnswer = UseTime);  // For 5D-Tensor
    Tensor(int pSize0, int pSize1, int pSize2, int pSize3, IsUseTime pAnswer = UseTime);  // For 4D-Tensor
    Tensor(int pSize0, int pSize1, int pSize2, IsUseTime pAnswer = UseTime);  // For 3D-Tensor
    Tensor(int pSize0, int pSize1, IsUseTime pAnswer = UseTime);  // For 2D-Tensor
    Tensor(int pSize0, IsUseTime pAnswer = UseTime);  // For 1D-Tensor
    Tensor(Shape *pShape, IsUseTime pAnswer = UseTime);
    Tensor(Tensor<DTYPE> *pTensor);  // Copy Constructor

    virtual ~Tensor();

    Shape                  * GetShape();
    Data<DTYPE>            * GetData();
    int                      GetCapacity();
    int                      GetTimeSize(); // 삭제 예정
    int                      GetBatchSize(); // 삭제 예정
    int                      GetChannelSize(); // 삭제 예정
    int                      GetRowSize(); // 삭제 예정
    int                      GetColSize(); // 삭제 예정
    DTYPE                  * GetCPUData(unsigned int pTime = 0);
    Device                   GetDevice();
    IsUseTime                GetIsUseTime();

    int                      Reshape(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize);

    void                     Reset();

    DTYPE                  & operator[](unsigned int index);

#ifdef __CUDNN__
    void                     SetDeviceCPU();
    void                     SetDeviceGPU();

    DTYPE                  * GetGPUData(unsigned int pTime = 0);

    cudnnTensorDescriptor_t& GetDescriptor();
    void                     Reset(cudnnHandle_t& pCudnnHandle);
#endif  // if __CUDNN__


    static Tensor<DTYPE>* Truncated_normal(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, float mean, float stddev);
    static Tensor<DTYPE>* Zeros(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize);
    static Tensor<DTYPE>* Constants(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, DTYPE constant);
};


inline unsigned int Index5D(Shape *pShape, int ti, int ba, int ch, int ro, int co) {
    return (((ti * (*pShape)[1] + ba) * (*pShape)[2] + ch) * (*pShape)[3] + ro) * (*pShape)[4] + co;
}

inline unsigned int Index4D(Shape *pShape, int ba, int ch, int ro, int co) {
    return ((ba * (*pShape)[2] + ch) * (*pShape)[3] + ro) * (*pShape)[4] + co;
}

inline unsigned int Index3D(Shape *pShape, int ch, int ro, int co) {
    return (ch * (*pShape)[3] + ro) * (*pShape)[4] + co;
}

inline unsigned int Index2D(Shape *pShape, int ro, int co) {
    return ro * (*pShape)[4] + co;
}

#endif  // TENSOR_H_
