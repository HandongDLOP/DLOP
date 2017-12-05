#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_    value

#include "Tensor.h"

template<typename DTYPE> class Operator;

enum OptimizeDirection {
    MAXIMIZE,
    MINIMIZE
};

template<typename DTYPE>
struct TrainableData {
    Tensor<DTYPE> *Data     = NULL;
    Tensor<DTYPE> *Gradient = NULL;
};

template<typename DTYPE>
class Optimizer {
private:
    /* data */
    // momentum이나 이런 애들은 따로 변수를 가지고 있어야 한다.

    Operator<DTYPE> *m_pObjectOperator = NULL;

    float m_LearningRate    = 0.f;
    int m_OptimizeDirection = 1;  // 1 or -1

    // int m_Batch = 0;

    TrainableData<DTYPE> **m_aTrainableData = NULL;
    int m_TrainableDataDegree               = 0;

public:
    Optimizer(Operator<DTYPE> *pObjectOperator, float pLearningRate, OptimizeDirection pOptimizeDirection) {
        std::cout << "Optimizer::Optimizer(Operator<DTYPE> *, float, OptimizeDirection)" << '\n';

        Alloc(pObjectOperator, pLearningRate, pOptimizeDirection);
    }

    virtual ~Optimizer() {
        std::cout << "Optimizer::~Optimizer()" << '\n';

        Delete();
    }

    int Alloc(Operator<DTYPE> *pObjectOperator, float pLearningRate, OptimizeDirection pOptimizeDirection) {
        SetObjectOperator(pObjectOperator);
        SetLearningRate(pLearningRate);
        SetOptimizeDirection(pOptimizeDirection);

        return 1;
    }

    int Delete() {
        for (int i = 0; i < m_TrainableDataDegree; i++) {
            delete m_aTrainableData[i];
        }
        delete m_aTrainableData;

        return 1;
    }

    int AddTrainableData(Tensor<DTYPE> *pData, Tensor<DTYPE> *pWeight) {
        if (m_TrainableDataDegree != 0) {
            TrainableData<DTYPE> **temp = new TrainableData<DTYPE> *[m_TrainableDataDegree + 1];
            std::copy(m_aTrainableData, m_aTrainableData + m_TrainableDataDegree, temp);

            delete[] m_aTrainableData;

            m_aTrainableData = temp;
        } else {
            m_aTrainableData = new TrainableData<DTYPE> *[m_TrainableDataDegree + 1];
        }

        TrainableData<DTYPE> *pTrainableData = new TrainableData<DTYPE>();
        pTrainableData->Data     = pData;
        pTrainableData->Gradient = pWeight;

        m_aTrainableData[m_TrainableDataDegree] = pTrainableData;

        m_TrainableDataDegree++;

        return 1;
    }

    int UpdateVariable() {
        for (int i = 0; i < m_TrainableDataDegree; i++) {
            // UpdateVariable(m_aTrainableData[i]->Data, m_aTrainableData[i]->Gradient);
            UpdateVariable(m_aTrainableData[i]);
        }
        return 1;
    }

    // virtual int UpdateVariable(Tensor<DTYPE> *Trainable, Tensor<DTYPE> *Gradient) = 0;
    virtual int UpdateVariable(TrainableData<DTYPE> *pTrainableData) = 0;


    void         SetObjectOperator(Operator<DTYPE> *pObjectOperator) {
        m_pObjectOperator = pObjectOperator;
    }

    void SetLearningRate(float pLearningRate) {
        m_LearningRate = pLearningRate;
    }

    // void SetBatch(int pBatch){
    // m_Batch = pBatch;
    // }

    void SetOptimizeDirection(OptimizeDirection pOptimizeDirection) {
        if (pOptimizeDirection == MAXIMIZE) m_OptimizeDirection = 1;
        else if (pOptimizeDirection == MINIMIZE) m_OptimizeDirection = -1;
    }

    Operator<DTYPE>* GetObjectOperator() const {
        return m_pObjectOperator;
    }

    float GetLearningRate() const {
        return m_LearningRate;
    }

    // int GetBatch(){
    // return m_Batch;
    // }

    int GetOptimizeDirection() {
        return m_OptimizeDirection;
    }
};

#endif  // OPTIMIZER_H_
