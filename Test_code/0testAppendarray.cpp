#include "iostream"
#include "algorithm"

int main(int argc, char const *argv[]) {
    int *a;

    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    a[3] = 4;
    a[4] = 5;
    a[5] = 7;

    for (int i = 0; i < 6; i++) {
        std::cout << a[i] << '\n';
    }

    int *temp = new int[6];
    std::copy(a, a + 5, temp);

    delete[] a;

    a = temp;
    a[5] = 6;

    for (int i = 0; i < 6; i++) {
        std::cout << a[i] << '\n';
    }

    return 0;
}
