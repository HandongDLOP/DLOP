#ifndef SHAPE_H_
#define SHAPE_H_    value

#define TRUE        1
#define FALSE       1


class Shape {
private:
    int m_Rank;
    int *m_Dimmension;

public:
    Shape() {}

    virtual ~Shape() {}

    int Alloc() {
        return TRUE;
    }
};


#endif  // SHAPE_H_
