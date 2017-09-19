#include "Operator.h"

/*
 *만약 NeuralNetwork가 가진 Operator의 주소가 input Layer일 경우
 */
bool Operator::PrePropagate() {
    this->Propagate();

    if (NextOperator != NULL) NextOperator->PrePropagate();

    return true;
}

bool Operator::PreBackPropagate() {
    if (NextOperator != NULL) NextOperator->PreBackPropagate();

    this->BackPropagate();

    return true;
}

/*
 *
 * 만약 NeuralNetwork가 가진 Operator의 주소가 Output Layer일 경우
 *
 * bool Operator::PropagateOrder() {
 *    // Postorder
 *
 *    if (NextOperator != NULL) NextOperator->PropagateOrder();
 *
 *    this->Propagate();
 *
 *    return true;
 * }
 *
 * bool Operator::BackPropagate() {
 *    // Preorder
 *
 *    this->BackPropagate();
 *
 *    if (NextOperator != NULL) NextOperator->BackPropagateOrder();
 *
 *    return true;
 * }
 *
 */
bool Propagate() {
    return true;
}

bool BackPropagate() {
    return true;
}
