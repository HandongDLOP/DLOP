#ifndef TENSOR_H_
#define TENSOR_H_

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <chrono>
#include <random>

// template <class T>

class Tensor {
private:
    // 현재는 scala 값은 따로 존재하지 않고 rnak0 dimension 1로 취급한다.
    int m_Rank;
    int *m_aShape;
    double *****m_aData;

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
        try{
            if(pRank == 5) Alloc(pShape[0], pShape[1], pShape[2], pShape[3], pShape[4]);
            // else if(pRank > 5) Alloc(pShape, pRank);
            else Alloc();
		} catch(...){
			printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
			exit(0);
		}
    }

    virtual ~Tensor() {
        std::cout << "Tensor::~Tensor()" << '\n';

        // delete를 제대로 하기 위해서는 계속해서 새로운 Tensor를 만들어낼 필요가 있다.
        // 추후 Delete를 고려해서 리펙토링 할 것
        Delete();
    }

    bool Alloc();
    bool Alloc(int pTime, int pBatch, int pChannel, int pRow, int pCol);
    // bool Alloc(int * pShape, int pRank);

    // bool Alloc(int pRank, std::initializer_list<int> pShape, INITIAL_MODE mode);

    bool Delete();

    // ===========================================================================================

    static Tensor* Truncated_normal(int pTime, int pBatch, int pChannel, int pRow, int pCol, double mean, double stddev);


    static Tensor* Zeros(int pTime, int pBatch, int pChannel, int pRow, int pCol);


    static Tensor* Constants(int pTime, int pBatch, int pChannel, int pRow, int pCol, double constant);


    // ===========================================================================================

    void SetData(double *****pData) {
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

    double***** GetData() const {
        return m_aData;
    }

    // ===========================================================================================

    void PrintData();
    void PrintShape();

    // Initialization(const std::string &type = "default");
};

#endif  // TENSOR_H_
