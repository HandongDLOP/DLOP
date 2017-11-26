#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_    value

#include <algorithm>

#include "Tensor.h"

struct TrainableData {
    Tensor *Data     = NULL;
    Tensor *Gradient = NULL;
};

class Optimizer {
private:
    /* data */
    // momentum이나 이런 애들은 따로 변수를 가지고 있어야 한다.

    TrainableData **m_aTrainableData = NULL;
    int m_TrainableDataDegree        = 0;
    float m_LearningRate             = 0.0;

public:
    Optimizer(float pLearningRate) {
        std::cout << "Optimizer::Optimizer()" << '\n';

        Alloc(pLearningRate);
    }

    virtual ~Optimizer() {
        std::cout << "Optimizer::~Optimizer()" << '\n';

        Delete();
    }

    bool Alloc(float pLearningRate) {
        m_LearningRate = pLearningRate;

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

    float GetLearningRate() {
        return m_LearningRate;
    }
};

#endif  // OPTIMIZER_H_
