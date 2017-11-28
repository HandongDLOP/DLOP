#include <iostream>

int main(int argc, char const *argv[]) {
    double sum[4][5] = {0.0};

    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 5; j++){
            std::cout << sum[i][j] << '\n';
        }
    }

    return 0;
}
