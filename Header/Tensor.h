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
public:
    typedef DTYPE *****TENSOR_DTYPE;

private:
    // 현재는 scala 값은 따로 존재하지 않고 rnak0 dimension 1로 취급한다.
    int m_Rank;
    int *m_aShape;
    DTYPE ***** m_aData;

public:
    Tensor() {
        std::cout << "Tensor::Tensor()" << '\n';
        Alloc();
    }

    Tensor(int pTime, int pBatch, int pChannel, int pRow, int pCol) {
        std::cout << "Tensor::Tensor(int, int, int, int, int)" << '\n';
        Alloc(pTime, pBatch, pChannel, pRow, pCol);
    }

    Tensor(int *pShape, int pRank = 5) {
        std::cout << "Tensor::Tensor(int, int, int, int, int)" << '\n';
        // 확장성을 위한 코드
        try {
            if (pRank == 5) Alloc(pShape[0], pShape[1], pShape[2], pShape[3], pShape[4]);
            // else if(pRank > 5) Alloc(pShape, pRank);
            else Alloc();
        } catch (...) {
            printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            exit(0);
        }
    }

    Tensor(TENSOR_DTYPE pData, int *pShape, int pRank = 5) {
        // std::cout << "Tensor::Tensor(TENSOR_DTYPE, int *, int)" << '\n';
        m_Rank   = pRank;
        m_aShape = pShape;
        m_aData  = pData;
    }

    virtual ~Tensor() {
        // std::cout << "Tensor::~Tensor()" << '\n';

        // delete를 제대로 하기 위해서는 계속해서 새로운 Tensor를 만들어낼 필요가 있다.
        Delete();
    }

    int Alloc();
    int Alloc(int pTime, int pBatch, int pChannel, int pRow, int pCol);
    // int Alloc(int * pShape, int pRank);

    // int Alloc(int pRank, std::initializer_list<int> pShape, INITIAL_MODE mode);

    int Delete();

    // ===========================================================================================

    static Tensor* Truncated_normal(int pTime, int pBatch, int pChannel, int pRow, int pCol, float mean, float stddev);


    static Tensor* Zeros(int pTime, int pBatch, int pChannel, int pRow, int pCol);


    static Tensor* Constants(int pTime, int pBatch, int pChannel, int pRow, int pCol, DTYPE constant);


    // ===========================================================================================

    void SetData(TENSOR_DTYPE pData) {
        if (m_aData != NULL) delete m_aData;
        m_aData = pData;
    }

    void Reset();

    // ===========================================================================================

    int GetRank() const {
        return m_Rank;
    }

    int* GetShape() const {
        return m_aShape;
    }

    int GetTime() const {
        return m_aShape[0];
    }

    int GetBatch() const {
        return m_aShape[1];
    }

    int GetChannel() const {
        return m_aShape[2];
    }

    int GetRow() const {
        return m_aShape[3];
    }

    int GetCol() const {
        return m_aShape[4];
    }

    TENSOR_DTYPE GetData() const {
        return m_aData;
    }

    //
    // typename GetType() const (
    // return TENSOR_DTYPE;
    // )

    // ===========================================================================================

    void PrintData(int forceprint = 0);
    void PrintShape();

    // Initialization(const std::string &type = "default");
};

#endif  // TENSOR_H_
