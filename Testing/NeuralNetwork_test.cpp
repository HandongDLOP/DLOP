#include <iostream>
#include <algorithm>

#include "..//Header//NeuralNetwork.h"

int main(int argc, char const *argv[]) {
    std::cout << "---------------Start-----------------" << '\n';

    NeuralNetwork HGUNN;

    // 입력데이터 만들기
    Tensor *var_1 = Tensor::Constants(1, 3, 0, 0, 0, 1.0);
    Operator *x = new Variable(var_1, "x", 0);

    Tensor *var_2 = Tensor::Truncated_normal(3, 3, 0, 0, 0, 0.0, 0.6);
    Operator *w = new Variable(var_2, "w", 1);

    Tensor *var_3 = Tensor::Constants(1, 3, 0, 0, 0, 1.0);
    Operator *b = new Variable(var_3, "b", 1); // 오류 발생 원인 찾기

    // 정답 데이터 만들기
    Tensor *var_4 = Tensor::Constants(1, 3, 0, 0, 0, 0.0);
    var_4->GetData()[0] = 0;
    var_4->GetData()[1] = 1;
    var_4->GetData()[2] = 0;
    Operator *answer = new Variable(var_4, "answer", 0);

    Operator *temp_1 = new MatMul(x, w);

    Operator *temp_2 = new Add(temp_1, b);

    // add_1->GetOutput()->PrintData();

    Operator *relu_1 = new Relu(temp_2, "relu_1");

    Operator *temp_4 = new MSE(relu_1, answer, "MSE");

    HGUNN.SetEndOperator(temp_4);

    // Optimizer 선택
    HGUNN.AllocOptimizer(STOCHASTIC_GRADIENT_DESCENT);

    // 학습 25번
    for (int i = 0; i < 20000; i++) {
        std::cout << "epoch : "<< i << '\n';
        HGUNN.Training();
    }


    return 0;
}
