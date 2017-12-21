#ifndef TENSOR_H_
#define TENSOR_H_

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <chrono>
#include <random>

template<typename DTYPE>
class Tensor {
private:
public:
    Tensor() {
        std::cout << "Tensor::Tensor()" << '\n';
    }

    Tensor(int pTime, int pBatch, int pChannel, int pRow, int pCol) {
        std::cout << "Tensor::Tensor(int, int, int, int, int)" << '\n';
    }

    Tensor(int *pShape, int pRank = 5) {
        std::cout << "Tensor::Tensor(int, int, int, int, int)" << '\n';
    }

    // Tensor(TENSOR_DTYPE pData, int *pShape, int pRank = 5) {}

    virtual ~Tensor() {
        Delete();
    }

    int Alloc();
    int Alloc(int pTime, int pBatch, int pChannel, int pRow, int pCol);


    int Delete();

    // ===========================================================================================

    static Tensor* Truncated_normal(int pTime, int pBatch, int pChannel, int pRow, int pCol, float mean, float stddev);


    static Tensor* Zeros(int pTime, int pBatch, int pChannel, int pRow, int pCol);


    static Tensor* Constants(int pTime, int pBatch, int pChannel, int pRow, int pCol, DTYPE constant);


    // ===========================================================================================

    // void SetData(TENSOR_DTYPE pData) {}

    void Reset();

    // ===========================================================================================

    // int          GetRank() const {}
    //
    // int        * GetShape() const {}
    //
    // int          GetTime() const {}
    //
    // int          GetBatch() const {}
    //
    // int          GetChannel() const {}
    //
    // int          GetRow() const {}
    //
    // int          GetCol() const {}

    // TENSOR_DTYPE GetData() const {}


    void PrintData(int forceprint = 0);
    void PrintShape();
};

#endif  // TENSOR_H_
