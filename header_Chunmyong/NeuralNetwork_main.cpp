/*
 * @breif: 학습이 되거나 복잡한 연산은 class 단에서 모두 마무리하고,
 *         main에서 detect하는 모든 Ark변수는 Variable 들이며,
 *         학습의 대상으로, 사용자가 실제로 얻고 싶은 결과들이다.
 *         이 것들을 저장하는 방법들에 대해서 생각할 필요가 있다.
 */
#include <iostream>

#include "DLOP.h"

int main(int argc, char const *argv[]) {
    std::cout << "---------------Start-----------------" << '\n';

    // NeuralNetwork HGUNN;
    //
    // /*
    //  * @breif: plaseholder는 Operator를 상속받으며, NeuralNetwork에서만 initialize 한다.
    //  */
    // Operator *x(input) = HGUNN.Placeholder("float", shape = [mini_batch, 28 * 28]);
    //
    // // ===========================<layer : Conv_1>============================
    // // =======================================================================
    //
    // /*
    //  * @breif: Static Method를 사용한다. (객체 없이 Method를 사용하는 것이 가능하다)
    //  */
    // Operator *W_conv1 = new Variable(truncated_normal({ 5, 5, 1, 32 }, stddev = 0.1));
    // Operator *b_conv1 = new Variable(constant(0.1, { 32 }));
    //
    // // ========================<With MetaParameter()>=========================
    //
    // /*
    //  * @breif: MetaParameter initialize with its Usage
    //  *         And initialize Operator
    //  *         Personally, We can delete MetaParameter()
    //  */
    // MetaParameter *pConvParam = new ConvParam(W_conv1, strides = { 1, 1, 1, 1 }, padding = "SAME");
    // Operator *Conv_1          = new Convolution(x, pConvParam);
    // // =======================================================================
    // // =======================<With no MetaParameter()>=======================
    //
    // /*
    //  * @breif: initialize Operator, and it can be work
    //  */
    // Operator *Conv_1 = new Convolution(x, W_conv1, , strides = { 1, 1, 1, 1 }, padding = "SAME");
    // // =======================================================================
    //
    // Operator *Relu_1 = new Relu(Conv_1, B_conv1);
    //
    // Operator *Pool_1 = new Maxpooling(Relu_1, ksize = { 1, 2, 2, 1 }, strides = { 1, 2, 2, 1 }, padding = "SAME");
    //
    //
    // // ===========================<layer : Conv_2>============================
    // // =======================================================================
    //
    // Operator *W_conv2 = new Variable(Ark.truncated_normal({ 5, 5, 32, 64 }, stddev = 0.1));
    // Operator *b_conv2 = new Variable(Ark.constant(0.1, { 64 }));
    //
    // // =======================<With no MetaParameter()>=======================
    //
    // /*
    //  * @breif: initialize Operator, and it can be work
    //  */
    // Operator *Conv_2 = new Convolution(Pool_1, W_conv2, strides = { 1, 1, 1, 1 }, padding = "SAME");
    // // =======================================================================
    //
    // Operator *Relu_2 = new Relu(Conv_2, B_conv2);
    //
    // Operator *Pool_2 = new Maxpooling(Relu_2, ksize = { 1, 2, 2, 1 }, strides = { 1, 2, 2, 1 }, padding = "SAME");
    //
    //
    // // ===========================<layer : FC_1>==============================
    // // =======================================================================
    //
    // // reshape 부분 생각해야 함
    //
    //
    // // ===========================<layer : FC_2>==============================
    // // =======================================================================
    //
    // // Softmax까지 완료
    //
    // // ====================<layer : Cross_Entropy>============================
    // // =======================================================================
    //
    // // 여기서 부터는 NeuralNetwork().Training or Testing에서 받는 시작 Operator variable을 전달 받을 수 있도록 해야 한다.
    //
    // // ========================<layer : Optimizer>============================
    // // =======================================================================
    //
    // // ========================<layer : Accuracy>=============================
    // // =======================================================================
    //
    // /*
    //  * @breif: 이 부분에서 나오는 accuracy는 Training할 때나, Testing때 출력할 수 있다.
    //  *         (사실 뭐든 출력할 수 있다) // 아마 이 것도 operator일 가능성이 높다.
    //  */
    // accuracy = NULL;
    //
    // /*
    //  * @breif: Graph가 올바로 만들어졌는지 체크 및 initialize 작업
    //  */
    // HGUNN.Create_Graph("initialize")
    //
    // // ========================<layer : Run Graph>============================
    // // =======================================================================
    //
    // /*
    //  * @breif: We can explicit the start Operator variable
    //  */
    //
    // /*
    //  * @breif: EPOCH = 10
    //  */
    // for (int i = 0; i < EPOCH; i++) {
    //     /*
    //      * @breif: BATCH = 100
    //      */
    //     for (int j = 0; j < BATCH; j++) {
    //         /*
    //          * @breif: mini_batch = 100
    //          */
    //         batch = train.next_batch(100);
    //
    //         /*
    //          * @breif: parm_1은 method의 행동을 결정한다.(Overridding)
    //          */
    //         HGUNN.Training(train_step, basket(x = batch[0], y_ = batch[1]));
    //     }
    //     batch = train_data.next_batch(100);
    //
    //     /*
    //      * @breif: parm_1은 출력하고 싶은 정보이다.(Overridding)
    //      *         여러개를 출력하게 하는 것도 가능하다.
    //      */
    //     std::cout << HGUNN.Testing(accuracy, basket(x = batch[0], y_ = batch[1])) << '\n';
    // }
    //
    // std::cout << HGUNN.Testing(accuracy, basket(x = test_data[0], y_ = test_data[1])) << '\n';
    //
    // // (if dinamic allocation) delete HGUNN; // not in this code

    std::cout << "---------------Done-----------------" << '\n';

    return 0;
}
