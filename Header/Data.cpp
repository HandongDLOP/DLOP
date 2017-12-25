#include "Data.h"

template class Data<int>;
template class Data<float>;
template class Data<double>;

template<typename DTYPE> int Data<DTYPE>::Alloc(unsigned int pSize) {
    // int rank = 0;
    m_Size = pSize;
    m_Cols = SIZEOFCOLS;

    if (m_Size % SIZEOFCOLS != 0) {
        m_Rows  = m_Size / SIZEOFCOLS + 1;
        m_aData = new DTYPE *[m_Rows];

        for (int i = 0; i < m_Rows; i++) {
            if (i != (m_Rows - 1)) {
                m_aData[i] = new DTYPE[SIZEOFCOLS];

                for (int j = 0; j < SIZEOFCOLS; j++) {
                    m_aData[i][j] = 0.f;
                }
            } else {
                int cols = m_Size % SIZEOFCOLS;

                m_aData[i] = new DTYPE[cols];

                for (int j = 0; j < cols; j++) {
                    m_aData[i][j] = 0.f;
                }
            }
        }
    } else {
        m_Rows  = m_Size / SIZEOFCOLS;
        m_aData = new DTYPE *[m_Rows];

        for (int i = 0; i < m_Rows; i++) {
            m_aData[i] = new DTYPE[SIZEOFCOLS];

            for (int j = 0; j < SIZEOFCOLS; j++) {
                m_aData[i][j] = 0.f;
            }
        }
    }

    return TRUE;
}

template<typename DTYPE> int Data<DTYPE>::Alloc(Data *pData) {
    m_Size = pData->GetSize();
    m_Cols = SIZEOFCOLS;

    if (m_Size % SIZEOFCOLS != 0) {
        m_Rows  = m_Size / SIZEOFCOLS + 1;
        m_aData = new DTYPE *[m_Rows];

        for (int i = 0; i < m_Rows; i++) {
            if (i != (m_Rows - 1)) {
                m_aData[i] = new DTYPE[SIZEOFCOLS];

                for (int j = 0; j < SIZEOFCOLS; j++) {
                    m_aData[i][j] = (*pData)[i * SIZEOFCOLS + j];
                }
            } else {
                int cols = m_Size % SIZEOFCOLS;

                m_aData[i] = new DTYPE[cols];

                for (int j = 0; j < cols; j++) {
                    m_aData[i][j] = (*pData)[i * SIZEOFCOLS + j];
                }
            }
        }
    } else {
        m_Rows  = m_Size / SIZEOFCOLS;
        m_aData = new DTYPE *[m_Rows];

        for (int i = 0; i < m_Rows; i++) {
            m_aData[i] = new DTYPE[SIZEOFCOLS];

            for (int j = 0; j < SIZEOFCOLS; j++) {
                m_aData[i][j] = (*pData)[i * SIZEOFCOLS + j];
            }
        }
    }

    return TRUE;
}

template<typename DTYPE> int Data<DTYPE>::Delete() {
    for (int i = 0; i < m_Rows; i++) {
        delete[] m_aData[i];
    }

    delete[] m_aData;

    return TRUE;
}

template<typename DTYPE> int Data<DTYPE>::GetSize() {
    return m_Size;
}

template<typename DTYPE> int Data<DTYPE>::GetCols() {
    return m_Cols;
}

template<typename DTYPE> int Data<DTYPE>::GetRows() {
    return m_Rows;
}

// example code
int main(int argc, char const *argv[]) {
    // Shape *pShape = new Shape(4, 1, 1, 1, 2048);

    Data<int> *pData = new Data<int>(2048);

    std::cout << (*pData)[2048] << '\n';
    // (*pData)[100000] = 1;

    // Data<int> *target = new Data<int>(pData);

    // std::cout << (*target)[100000] << '\n';

    // Caution: Data Class cannot deallocate dynamic Shape value.
    // delete pShape;
    delete pData;
    // delete target;

    return 0;
}
