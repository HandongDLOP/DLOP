#ifndef TENSOR_H_
#define TENSOR_H_

#include <time.h>
#include <math.h>
// #include <chrono>
// #include <random>

#include "Data.h"

template<typename DTYPE> class Tensor {
private:
    Shape *m_aShape;
    Data<DTYPE> *m_aData;

public:
    Tensor();
    Tensor(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize);  // For 5D-Tensor
    virtual ~Tensor();

    int          Alloc(Shape *pShape);
    void         Delete();

    Shape      * GetShape();
    Data<DTYPE>* GetData();

    int          GetTimeSize();
    int          GetBatchSize();
    int          GetChannelSize();
    int          GetRowSize();
    int          GetColSize();

    ///////////////////////////////////////////////////////////////////

    DTYPE& operator[](unsigned int index);
    // DTYPE& GetDatum(int ti, int ba, int ch, int ro, int co);

    ///////////////////////////////////////////////////////////////////

    static Tensor* Truncated_normal(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, float mean, float stddev);

    static Tensor* Zeros(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize);

    static Tensor* Constants(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, DTYPE constant);
};

///////////////////////////////////////////////////////////////////

inline unsigned int Index5D(Shape *pShape, int ti, int ba, int ch, int ro, int co);

#endif  // TENSOR_H_
