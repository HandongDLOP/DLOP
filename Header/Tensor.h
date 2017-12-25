#ifndef TENSOR_H_
#define TENSOR_H_

#include <time.h>
#include <math.h>
#include <chrono>
#include <random>

#include "Data.h"

template<typename DTYPE> class Tensor {
private:
    Shape *m_aShape;
    Data<DTYPE> *m_aData;

public:
    Tensor() {
        std::cout << "Tensor::Tensor()" << '\n';
    }

    Tensor(Shape *pShape) {
        std::cout << "Tensor::Tensor(Shape*)" << '\n';
        Alloc(pShape);
    }

    virtual ~Tensor() {
        Delete();
    }

    int Alloc(Shape *pShape);
    int Delete();

    ///////////////////////////////////////////////////////////////////

    // void SetData(TENSOR_DTYPE pData) {}

    // void SetData(Shape *pShape);

    unsigned int Index(int rank, ...);

    ///////////////////////////////////////////////////////////////////

    static Tensor* Truncated_normal(Shape *pShape, float mean, float stddev);

    static Tensor* Zeros(Shape *pShape);

    static Tensor* Constants(Shape *pShape, DTYPE constant);

    ///////////////////////////////////////////////////////////////////
};

#endif  // TENSOR_H_
