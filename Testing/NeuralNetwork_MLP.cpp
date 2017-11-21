#include <iostream>
#include <string>

#include "..//Header//NeuralNetwork.h"


int main(int argc, char const *argv[]) {
    std::cout << "---------------Start-----------------" << '\n';

    NeuralNetwork HGUNN;

    // create input data
    Tensor   *_x1 = Tensor::Constants(1, 2, 0, 0, 0, 1.0);
    Operator *x1  = HGUNN.AddPlaceholder(_x1, "x1");

    // create label data
    Tensor   *_ans = Tensor::Constants(1, 2, 0, 0, 0, 0.0);
    Operator *ans  = HGUNN.AddPlaceholder(_ans, "answer");
    ans->GetOutput()->GetData()[0] = 0;
    ans->GetOutput()->GetData()[1] = 1;

    // ======================= layer 1=======================
    Tensor   *_w1 = Tensor::Truncated_normal(2, 2, 0, 0, 0, 0.0, 0.6);
    Operator *w1  = new Variable(_w1, "w1", 1);

    Tensor   *_b1 = Tensor::Constants(1, 2, 0, 0, 0, 1.0);
    Operator *b1  = new Variable(_b1, "b1", 1); // 오류 발생 원인 찾기

    Operator *mat_1 = new MatMul(x1, w1);

    Operator *add_1 = new Add(mat_1, b1);

    Operator *sig_1 = new Sigmoid(add_1, "sig_1");

    // ======================= layer 2=======================
    Tensor   *_w2 = Tensor::Truncated_normal(2, 2, 0, 0, 0, 0.0, 0.6);
    Operator *w2  = new Variable(_w2, "w2", 1);

    Tensor   *_b2 = Tensor::Constants(1, 2, 0, 0, 0, 1.0);
    Operator *b2  = new Variable(_b2, "b2", 1); // 오류 발생 원인 찾기

    Operator *mat_2 = new MatMul(sig_1, w2);

    Operator *add_2 = new Add(mat_2, b2);

    Operator *sig_2 = new Sigmoid(add_2, "sig_2");

    Operator *err = new MSE(sig_2, ans, "MSE");

    // ======================= Create Graph =======================

    HGUNN.CreateGraph(STOCHASTIC_GRADIENT_DESCENT, err);

    // ======================= Training =======================

    if(argc != 2){
        std::cout << "There is no count of training" << '\n';
        return 0;
    }

    for (int i = 0; i < atoi(argv[1]); i++) {
        std::cout << "epoch : " << i << '\n';

        if (i % 4 == 0) {
            x1->GetOutput()->GetData()[0] = 0;
            x1->GetOutput()->GetData()[1] = 0;

            ans->GetOutput()->GetData()[0] = 1;
            ans->GetOutput()->GetData()[1] = 0;
        }

        if (i % 4 == 1) {
            x1->GetOutput()->GetData()[0] = 1;
            x1->GetOutput()->GetData()[1] = 0;

            ans->GetOutput()->GetData()[0] = 0;
            ans->GetOutput()->GetData()[1] = 1;
        }

        if (i % 4 == 2) {
            x1->GetOutput()->GetData()[0] = 0;
            x1->GetOutput()->GetData()[1] = 1;

            ans->GetOutput()->GetData()[0] = 0;
            ans->GetOutput()->GetData()[1] = 1;
        }

        if (i % 4 == 3) {
            x1->GetOutput()->GetData()[0] = 1;
            x1->GetOutput()->GetData()[1] = 1;

            ans->GetOutput()->GetData()[0] = 1;
            ans->GetOutput()->GetData()[1] = 0;
        }

        HGUNN.Training();
    }

    // ======================= Testing =======================

    for(int i =0 ; i < 4; i++){
        std::cout << "input : " << i << '\n';

        if (i % 4 == 0) {
            x1->GetOutput()->GetData()[0] = 0;
            x1->GetOutput()->GetData()[1] = 0;
        }

        if (i % 4 == 1) {
            x1->GetOutput()->GetData()[0] = 1;
            x1->GetOutput()->GetData()[1] = 0;
        }

        if (i % 4 == 2) {
            x1->GetOutput()->GetData()[0] = 0;
            x1->GetOutput()->GetData()[1] = 1;
        }

        if (i % 4 == 3) {
            x1->GetOutput()->GetData()[0] = 1;
            x1->GetOutput()->GetData()[1] = 1;
        }

        HGUNN.Testing();
    }


    std::cout << "---------------End-----------------" << '\n';
    return 0;
}
