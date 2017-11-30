#include "Tensor.h"

bool Tensor::Alloc() {
    m_Rank   = 0;
    m_aShape = NULL;
    m_aData  = NULL;

    return true;
}

bool Tensor::Alloc(int pTime, int pBatch, int pChannel, int pRow, int pCol) {
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

    m_aData = new double ****[pTime];

    for (int ti = 0; ti < pTime; ti++) {
        m_aData[ti] = new double ***[pBatch];

        for (int ba = 0; ba < pBatch; ba++) {
            m_aData[ti][ba] = new double **[pChannel];

            for (int ch = 0; ch < pChannel; ch++) {
                m_aData[ti][ba][ch] = new double *[pRow];

                for (int ro = 0; ro < pRow; ro++) {
                    m_aData[ti][ba][ch][ro] = new double[pCol];

                    for (int co = 0; co < pCol; co++) {
                        m_aData[ti][ba][ch][ro][co] = 0.0;  // 0으로 초기화
                    }
                }
            }
        }
    }

    // ==============================================================


    return true;
}

bool Tensor::Delete() {
    // std::cout << "Tensor::Delete()" << '\n';

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

    return true;
}

void Tensor::Reset() {
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

Tensor * Tensor::Truncated_normal(int pTime, int pBatch, int pChannel, int pRow, int pCol, double mean, double stddev) {
    std::cout << "Tensor::Truncated_normal()" << '\n';

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> rand(mean, stddev);

    //// 추후 교수님이 주신 코드를 참고해서 바꿀 것
    // double   stdev = (double)sqrt(2.F / (pRow + pCol + pChannel));
    // unsigned seed  = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
    // std::default_random_engine generator(seed);
    // std::normal_distribution<double> dist(0.F, stdev);

    Tensor *temp_Tensor   = new Tensor(pTime, pBatch, pChannel, pRow, pCol);
    double *****temp_data = temp_Tensor->GetData();

    for (int ti = 0; ti < pTime; ti++) {
        for (int ba = 0; ba < pBatch; ba++) {
            for (int ch = 0; ch < pChannel; ch++) {
                for (int ro = 0; ro < pRow; ro++) {
                    for (int co = 0; co < pCol; co++) {
                        // temp_data[ti][ba][ch][ro][co] = (double)dist(generator);
                        temp_data[ti][ba][ch][ro][co] = rand(gen);
                    }
                }
            }
        }
    }

    return temp_Tensor;
}

Tensor * Tensor::Zeros(int pTime, int pBatch, int pChannel, int pRow, int pCol) {
    std::cout << "Tensor::Zero()" << '\n';

    Tensor *temp_Tensor = new Tensor(pTime, pBatch, pChannel, pRow, pCol);

    return temp_Tensor;
}

Tensor * Tensor::Constants(int pTime, int pBatch, int pChannel, int pRow, int pCol, double constant) {
    std::cout << "Tensor::Constant()" << '\n';

    Tensor *temp_Tensor   = new Tensor(pTime, pBatch, pChannel, pRow, pCol);
    double *****temp_data = temp_Tensor->GetData();

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

void Tensor::PrintData(int forceprint) {
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
                        std::cout << " ]";
                    }
                    std::cout << " ]";
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

void Tensor::PrintShape() {
    std::cout << "[ ";

    for (int ra = 0; ra < m_Rank; ra++) {
        std::cout << m_aShape[ra] << ", ";
    }
    std::cout << " ]" << '\n';
}
