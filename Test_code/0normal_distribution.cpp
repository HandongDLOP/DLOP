#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "math.h"

#define DIST 100

float gaussianRandom(float average, float stdev);

int   main(int argc, char const *argv[]) {
    srand((unsigned)time(NULL));
    float temp   = 0.f;
    int   approx = 0;

    int count[DIST] = { 0, };

    for (int i = 1; i <= 1000; i++) {
        // 평균은 0.0 이고, 표준편차는 0.1 의 경우
        temp = gaussianRandom(50.0, 12.0);
        printf("%.17f\n", temp);

        if (temp >= 0) {
            approx = temp / 1;
            count[approx]++;
        }
    }

    printf("\n");

    for (int i = 0; i < DIST; i++) {
        printf("|||");
        for (int j = 0; j < count[i]; j++) {
            printf("*");
        }
        printf("\n");
    }

    return 0;
}

float gaussianRandom(float average, float stdev) {
    float v1, v2, s, temp;

    do {
        v1 = 2 * ((float)rand() / RAND_MAX) - 1;  // -1.0 ~ 1.0 까지의 값
        v2 = 2 * ((float)rand() / RAND_MAX) - 1;  // -1.0 ~ 1.0 까지의 값
        s  = v1 * v1 + v2 * v2;
    } while (s >= 1 || s == 0);

    s = sqrt((-2 * log(s)) / s);

    temp = v1 * s;
    temp = (stdev * temp) + average;

    return temp;
}
