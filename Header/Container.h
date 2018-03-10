#include "Common.h"

template<typename DTYPE> class Container {
private:
    DTYPE *m_aElement;
    int m_size;

public:
    Container();
    virtual ~Container();

    int Append(DTYPE pElement);

    int GetSize() {
        return m_size;
    }

    DTYPE GetLast() {
        return m_aElement[m_size - 1];
    }

    DTYPE operator[](unsigned int index);
};
