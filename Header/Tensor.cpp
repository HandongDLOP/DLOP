#include "Tensor.h"

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;

template<typename DTYPE>
int Tensor<DTYPE>::Alloc() {
    m_Rank   = 0;
    m_aShape = NULL;
    m_aData  = NULL;

    return 1;
}

template<typename DTYPE>
int Tensor<DTYPE>::Alloc(int pTime, int pBatch, int pChannel, int pRow, int pCol) {
    // ============================================================

    m_Rank = 5;

    // ============================================================

    m_aShape = new int[5];

    m_aShape[0] = pTime;
    m_aShape[1] = pBatch;
    m_aShape[2] = pChannel;
    m_aShape[3] = pRow;
    m_aShape[4] = pCol;

    // =============================================================

    m_aData = new DTYPE * ** *[pTime];

    for (int ti = 0; ti < pTime; ti++) {
        m_aData[ti] = new DTYPE * * *[pBatch];

        for (int ba = 0; ba < pBatch; ba++) {
            m_aData[ti][ba] = new DTYPE * *[pChannel];

            for (int ch = 0; ch < pChannel; ch++) {
                m_aData[ti][ba][ch] = new DTYPE *[pRow];

                for (int ro = 0; ro < pRow; ro++) {
                    m_aData[ti][ba][ch][ro] = new DTYPE[pCol];

                    for (int co = 0; co < pCol; co++) {
                        m_aData[ti][ba][ch][ro][co] = 0.0;  // 0으로 초기화
                    }
                }
            }
        }
    }

    // ==============================================================


    return 1;
}

template<typename DTYPE>
int Tensor<DTYPE>::Delete() {
    // std::cout << "Tensor<DTYPE>::Delete()" << '\n';

    int Time    = GetTime();
    int Batch   = GetBatch();
    int Channel = GetChannel();
    int Row     = GetRow();

    // int Col     = m_aShape[4];

    // =============================================================

    for (int ti = 0; ti < Time; ti++) {
        for (int ba = 0; ba < Batch; ba++) {
            for (int ch = 0; ch < Channel; ch++) {
                for (int ro = 0; ro < Row; ro++) {
                    delete[] m_aData[ti][ba][ch][ro];
                }
                delete[] m_aData[ti][ba][ch];
            }
            delete[] m_aData[ti][ba];
        }
        delete[] m_aData[ti];
    }
    delete[] m_aData;

    // =============================================================

    delete[] m_aShape;

    return 1;
}

template<typename DTYPE>
void Tensor<DTYPE>::Reset() {
    for (int ti = 0; ti < m_aShape[0]; ti++) {
        for (int ba = 0; ba < m_aShape[1]; ba++) {
            for (int ch = 0; ch < m_aShape[2]; ch++) {
                for (int ro = 0; ro < m_aShape[3]; ro++) {
                    for (int co = 0; co < m_aShape[4]; co++) {
                        m_aData[ti][ba][ch][ro][co] = 0;
                    }
                }
            }
        }
    }
}

// ===========================================================================================

template<typename DTYPE>
Tensor<DTYPE> *Tensor<DTYPE>::Truncated_normal(int pTime, int pBatch, int pChannel, int pRow, int pCol, float mean, float stddev) {
    std::cout << "Tensor<DTYPE>::Truncated_normal()" << '\n';

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> rand(mean, stddev);

    //// 추후 교수님이 주신 코드를 참고해서 바꿀 것
    // DTYPE   stdev = (DTYPE)sqrt(2.F / (pRow + pCol + pChannel));
    // unsigned seed  = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
    // std::default_random_engine generator(seed);
    // std::normal_distribution<DTYPE> dist(0.F, stdev);

    Tensor<DTYPE> *temp_Tensor = new Tensor(pTime, pBatch, pChannel, pRow, pCol);
    TENSOR_DTYPE temp_data   = temp_Tensor->GetData();

    for (int ti = 0; ti < pTime; ti++) {
        for (int ba = 0; ba < pBatch; ba++) {
            for (int ch = 0; ch < pChannel; ch++) {
                for (int ro = 0; ro < pRow; ro++) {
                    for (int co = 0; co < pCol; co++) {
                        // temp_data[ti][ba][ch][ro][co] = (DTYPE)dist(generator);
                        temp_data[ti][ba][ch][ro][co] = rand(gen);
                    }
                }
            }
        }
    }

    return temp_Tensor;
}

template<typename DTYPE>
Tensor<DTYPE> *Tensor<DTYPE>::Zeros(int pTime, int pBatch, int pChannel, int pRow, int pCol) {
    std::cout << "Tensor<DTYPE>::Zero()" << '\n';

    Tensor<DTYPE> *temp_Tensor = new Tensor(pTime, pBatch, pChannel, pRow, pCol);

    return temp_Tensor;
}

template<typename DTYPE>
Tensor<DTYPE> *Tensor<DTYPE>::Constants(int pTime, int pBatch, int pChannel, int pRow, int pCol, DTYPE constant) {
    std::cout << "Tensor<DTYPE>::Constant()" << '\n';

    Tensor<DTYPE> *temp_Tensor = new Tensor(pTime, pBatch, pChannel, pRow, pCol);
    TENSOR_DTYPE temp_data   = temp_Tensor->GetData();

    for (int ti = 0; ti < pTime; ti++) {
        for (int ba = 0; ba < pBatch; ba++) {
            for (int ch = 0; ch < pChannel; ch++) {
                for (int ro = 0; ro < pRow; ro++) {
                    for (int co = 0; co < pCol; co++) {
                        temp_data[ti][ba][ch][ro][co] = constant;
                    }
                }
            }
        }
    }

    return temp_Tensor;
}

template<typename DTYPE>
void Tensor<DTYPE>::PrintData(int forceprint) {
    if (m_aData == NULL) {
        std::cout << "data is empty!" << '\n';
        exit(0);
    }

    int Time    = m_aShape[0];
    int Batch   = m_aShape[1];
    int Channel = m_aShape[2];
    int Row     = m_aShape[3];
    int Col     = m_aShape[4];

    int lenght = Time * Batch * Channel * Row * Col;

    if ((lenght < 100) || (forceprint == 1)) {
        std::cout << "[ ";

        for (int ti = 0; ti < Time; ti++) {
            std::cout << "[ \n";

            for (int ba = 0; ba < Batch; ba++) {
                std::cout << "[ ";

                for (int ch = 0; ch < Channel; ch++) {
                    std::cout << "[ ";

                    for (int ro = 0; ro < Row; ro++) {
                        std::cout << "[ ";

                        for (int co = 0; co < Col; co++) {
                            std::cout << m_aData[ti][ba][ch][ro][co] << ", ";
                        }
                        std::cout << " ]\n";
                    }
                    std::cout << " ]\n";
                }
                std::cout << " ]\n";
            }
            std::cout << " ]";
        }
        std::cout << " ]\n";
    } else {
        std::cout << "too big!" << '\n';
    }
}

template<typename DTYPE>
void Tensor<DTYPE>::PrintShape() {
    std::cout << "[ ";

    for (int ra = 0; ra < m_Rank; ra++) {
        std::cout << m_aShape[ra] << ", ";
    }
    std::cout << " ]" << '\n';
}
