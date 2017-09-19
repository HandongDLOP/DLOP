#include "Operator.h"

///*
// *만약 NeuralNetwork가 가진 Operator의 주소가 input Layer일 경우
// */
// bool Operator::PrePropagate() {
// this->Propagate();
//
// if (NextOperator != NULL) NextOperator->PrePropagate();
//
// return true;
// }
//
// bool Operator::PreBackPropagate() {
// if (NextOperator != NULL) NextOperator->PreBackPropagate();
//
// this->BackPropagate();
//
// return true;
// }


// 만약 NeuralNetwork가 가진 Operator의 주소가 Output Layer일 경우

bool Operator::ForwardPropagate() {
    // Postorder
    if (NextOperator != NULL) NextOperator->ForwardPropagate();

    this->ExcuteForwardPropagate();

    return true;
}

bool Operator::BackPropagate() {
    // Preorder
    this->ExcuteBackPropagate();

    if (NextOperator != NULL) NextOperator->BackPropagate();

    return true;
}

bool Operator::ExcuteForwardPropagate() {

    return true;
}

bool Operator::ExcuteBackPropagate() {
    return true;
}
