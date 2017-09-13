#ifndef Tensor_H_
#define Tensor_H_

#include <iostream>
#include <string>

class Tensor {
private:
    int dimension;
    char type;  // 추후 수정 예정
    void *data;

public:
    Tensor();
    virtual ~Tensor();

    //Initialization(const std::string &type = "default");
};

#endif  // Tensor_H_
