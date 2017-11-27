#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_    value

#include <algorithm>

#include "Tensor.h"

class Operator;

enum OptimizeDirection {
    MAXIMIZE,
    MINIMIZE
};

struct TrainableData {
    Tensor *Data     = NULL;
    Tensor *Gradient = NULL;
};

class Optimizer {
private:
    /* data */
    // momentum이나 이런 애들은 따로 변수를 가지고 있어야 한다.

    Operator *ObjectOperator = NULL;

    float m_LearningRate    = 0.f;
    int m_OptimizeDirection = 1;  // 1 or -1

    TrainableData **m_aTrainableData = NULL;
    int m_TrainableDataDegree        = 0;

public:
    Optimizer(Operator *pObjectOperator, float pLearningRate, OptimizeDirection pOptimizeDirection) {
        std::cout << "Optimizer::Optimizer(Operator *, float, OptimizeDirection)" << '\n';

        Alloc(pObjectOperator, pLearningRate, pOptimizeDirection);
    }

    virtual ~Optimizer() {
        std::cout << "Optimizer::~Optimizer()" << '\n';

        Delete();
    }

    bool Alloc(Operator *pObjectOperator, float pLearningRate, OptimizeDirection pOptimizeDirection) {
        ObjectOperator = pObjectOperator;
        m_LearningRate = pLearningRate;
        SetOptimizeDirection(pOptimizeDirection);

        return true;
    }

    bool Delete() {
        for (int i = 0; i < m_TrainableDataDegree; i++) {
            delete m_aTrainableData[i];
        }
        delete m_aTrainableData;

        return true;
    }

    bool AddTrainableData(Tensor *pData, Tensor *pWeight) {
        if (m_TrainableDataDegree != 0) {
            TrainableData **temp = new TrainableData *[m_TrainableDataDegree + 1];
            std::copy(m_aTrainableData, m_aTrainableData + m_TrainableDataDegree, temp);

            delete[] m_aTrainableData;

            m_aTrainableData = temp;
        } else {
            m_aTrainableData = new TrainableData *[m_TrainableDataDegree + 1];
        }

        TrainableData *pTrainableData = new TrainableData();
        pTrainableData->Data     = pData;
        pTrainableData->Gradient = pWeight;

        m_aTrainableData[m_TrainableDataDegree] = pTrainableData;

        m_TrainableDataDegree++;

        return true;
    }

    bool UpdateWeight() {
        for (int i = 0; i < m_TrainableDataDegree; i++) {
            // UpdateWeight(m_aTrainableData[i]->Data, m_aTrainableData[i]->Gradient);
            UpdateWeight(m_aTrainableData[i]);
        }
        return true;
    }

    // virtual bool UpdateWeight(Tensor *Trainable, Tensor *Gradient) = 0;
    virtual bool UpdateWeight(TrainableData *pTrainableData) = 0;


    void         SetLearningRate(float pLearningRate) {
        m_LearningRate = pLearningRate;
    }

    void SetOptimizeDirection(OptimizeDirection pOptimizeDirection) {
        if (pOptimizeDirection == MAXIMIZE) m_OptimizeDirection = 1;
        else if (pOptimizeDirection == MINIMIZE) m_OptimizeDirection = -1;
    }

    float GetLearningRate() {
        return m_LearningRate;
    }

    int GetOptimizeDirection(){
        return m_OptimizeDirection;
    }
};

#endif  // OPTIMIZER_H_
