#include <iostream>

template <typename T, typename DTYPE>
DTYPE Cal2(T param1, DTYPE param2){
    return param1[0][0][0] + param2;
}


template <typename T>
T Cal(T param1, T param2){
    param1[0][0][0][0][0] += param2[0][0][0][0][0];

    return param1;

}

int main(int argc, char const *argv[]) {

    int temp1[1][1][1][1][1] = {{{{{2}}}}};
    int temp2[1][1][1][1][1] = {{{{{5}}}}};

    int temp3 = Cal2(temp1[0][0], temp2[0][0][0][0][0]);

    std::cout << temp3 << '\n';

    return 0;
}
