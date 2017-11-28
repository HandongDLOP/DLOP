#include <iostream>
#include <time.h>

#define BATCH    100

int main(int argc, char const *argv[]) {
    srand(time(NULL));

    for (int i = 0; i < BATCH; i++) {
        std::cout << rand()%150 << '\n';
    }

    return 0;
}
