#ifndef FACTORYMETHOD_H_
#define FACTORYMETHOD_H_    value

#include <string>
#include <iostream>
#include "Layer.h"
#include "Activation.h"
#include "Objective.h"
#include "Operator.h"

// factory method를 사용하는 class
class FactoryMethod {
private:
    /* data */

public:
    FactoryMethod();
    virtual ~FactoryMethod();

    // // Parameter std::string type에 따라 Activation 자식 클래스 생성자 반환
    // static Layer    * LayerFactory(const std::string& type);
    // // Parameter std::string type에 따라 Objective 자식 클래스 생성자 반환
    // static Objective* ObjectiveFactory(const std::string& type);
    // // Parameter std::string type에 따라 Operator 자식 클래스 생성자 반환
    // static Operator * OperatorFactory(const std::string& type);
};
#endif  // FACTORYMETHOD_H_
