#ifndef Tensor_H_
#define Tensor_H_

class Tensor {
private:
    int dimension;
    char type;  // 추후 수정 예정
    void *data;

public:
    Tensor();
    virtual ~Tensor();
};

#endif  // Tensor_H_
