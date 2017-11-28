#include <fstream>
#include <iostream>
#include <string>
#include <time.h>

#include "..//Header//NeuralNetwork.h"

#define DIMENSION_OF_MNIST_IMAGE    784

#define NUMBER_OF_TRAIN_DATA        60000
#define NUMBER_OF_TEST_DATA         10000

#define TEST_IMAGE_FILE             "data/t10k-images-idx3-ubyte"
#define TEST_LABEL_FILE             "data/t10k-labels-idx1-ubyte"
#define TRAIN_IMAGE_FILE            "data/train-images-idx3-ubyte"
#define TRAIN_LABEL_FILE            "data/train-labels-idx1-ubyte"

using namespace std;

enum OPTION {
    TEST,
    TESTIMAGE,
    TESTLABEL,
    TRAIN,
    TRAINIMAGE,
    TRAINLABEL,
    DEFAULT
};

class DataSet {
private:
    // 직접 해제
    double **Test_image  = NULL;
    double **Test_label  = NULL;
    double **Train_image = NULL;
    double **Train_label = NULL;

    // 따로 해제
    Tensor *Test_image_feed  = NULL;
    Tensor *Test_label_feed  = NULL;
    Tensor *Train_image_feed = NULL;
    Tensor *Train_label_feed = NULL;

public:
    DataSet() {}

    virtual ~DataSet() {
        Delete();
    }

    void Delete() {
        for (int num = 0; num < NUMBER_OF_TEST_DATA; num++) {
            delete[] Test_image[num];
            delete[] Test_label[num];
        }
        delete Test_image;
        delete Test_label;

        for (int num = 0; num < NUMBER_OF_TRAIN_DATA; num++) {
            delete[] Train_image[num];
            delete[] Train_label[num];
        }
        delete Train_image;
        delete Train_label;
    }

    void CreateDataPair(OPTION pOption, int batch_size) {
        int number_of_data    = 0;
        int random            = 0;
        double **origin_image = NULL;
        double **origin_label = NULL;

        if (pOption == TEST) {
            origin_image   = Test_image;
            origin_label   = Test_label;
            number_of_data = NUMBER_OF_TEST_DATA;
        } else if (pOption == TRAIN) {
            origin_image   = Train_image;
            origin_label   = Train_label;
            number_of_data = NUMBER_OF_TRAIN_DATA;
        } else {
            std::cout << "invalid OPTION!" << '\n';
            exit(0);
        }

        // create input image data
        double *****image_data = new double ****[1];
        image_data[0] = new double ***[batch_size];

        int *image_shape = new int[5] { 1, batch_size, 1, 1, DIMENSION_OF_MNIST_IMAGE };
        int  image_rank  = 5;
        // double **origin_image = ReshapeData(image_option);

        // create input label data
        double *****label_data = new double ****[1];
        label_data[0] = new double ***[batch_size];

        int *label_shape = new int[5] { 1, batch_size, 1, 1, 10 };
        int  label_rank  = 5;
        // double **origin_label = ReshapeData(label_option);

        // 무작위 선택
        srand(time(NULL));

        for (int ba = 0; ba < batch_size; ba++) {
            random = rand() % number_of_data;

            image_data[0][ba]    = new double **[1];
            image_data[0][ba][0] = new double *[1];
            // image_data[0][ba][0][0] = origin_image[random];
            image_data[0][ba][0][0] = new double[DIMENSION_OF_MNIST_IMAGE];

            for (int dim = 0; dim < DIMENSION_OF_MNIST_IMAGE; dim++) {
                image_data[0][ba][0][0][dim] = origin_image[random][dim];
            }

            label_data[0][ba]       = new double **[1];
            label_data[0][ba][0]    = new double *[1];
            label_data[0][ba][0][0] = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            // std::cout << (int)origin_label[random][0] << '\n';
            label_data[0][ba][0][0][(int)origin_label[random][0]] = 1.0;
        }

        Tensor *image_Tensor = new Tensor(image_data, image_shape, image_rank);
        Tensor *label_Tensor = new Tensor(label_data, label_shape, label_rank);

        if (pOption == TEST) {
            Test_image_feed = image_Tensor;
            Test_label_feed = label_Tensor;
        } else if (pOption == TRAIN) {
            Train_image_feed = image_Tensor;
            Train_label_feed = label_Tensor;
        } else {
            std::cout << "invalid OPTION!" << '\n';
            exit(0);
        }
    }

    void SetTestImage(double **pTest_image) {
        Test_image = pTest_image;
    }

    void SetTestLabel(double **pTest_label) {
        Test_label = pTest_label;
    }

    void SetTrainImage(double **pTrain_image) {
        Train_image = pTrain_image;
    }

    void SetTrainLabel(double **pTrain_label) {
        Train_label = pTrain_label;
    }

    Tensor* GetFeedImage(OPTION pOption) {
        if (pOption == TEST) return Test_image_feed;
        else if (pOption == TRAIN) return Train_image_feed;
        else return NULL;
    }

    Tensor* GetFeedLabel(OPTION pOption) {
        if (pOption == TEST) return Test_label_feed;
        else if (pOption == TRAIN) return Train_label_feed;
        else return NULL;
    }
};

int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;

    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void IMAGE_Reader(string DATAPATH, double **arr) {
    // arr.resize(NumberOfImages, vector<double>(DataOfAnImage));
    ifstream fin;

    fin.open(DATAPATH, ios::binary);

    if (fin.is_open()) {
        int magic_number     = 0;
        int number_of_images = 0;
        int n_rows           = 0;
        int n_cols           = 0;

        fin.read((char *)&magic_number,     sizeof(magic_number));
        magic_number = ReverseInt(magic_number);

        fin.read((char *)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        fin.read((char *)&n_rows,           sizeof(n_rows));
        n_rows = ReverseInt(n_rows);

        fin.read((char *)&n_cols,           sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        int dim_of_image = n_rows * n_cols;

        for (int i = 0; i < number_of_images; ++i) {
            arr[i] = new double[dim_of_image];

            for (int d = 0; d < dim_of_image; ++d) {
                unsigned char temp = 0;
                fin.read((char *)&temp, sizeof(temp));
                arr[i][d] = (double)temp;
                // std::cout << arr[i][d] << ' ';
            }
            // std::cout << "\n\n";
        }
    }
}

void LABEL_Reader(string DATAPATH, double **arr) {
    // arr.resize(NumberOfImages, vector<double>(DataOfAnImage));
    ifstream fin;

    fin.open(DATAPATH, ios::binary);

    if (fin.is_open()) {
        int magic_number     = 0;
        int number_of_labels = 0;

        fin.read((char *)&magic_number,     sizeof(magic_number));
        magic_number = ReverseInt(magic_number);

        fin.read((char *)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = ReverseInt(number_of_labels);

        // std::cout << number_of_labels << '\n';

        for (int i = 0; i < number_of_labels; ++i) {
            arr[i] = new double[1];

            unsigned char temp = 0;
            fin.read((char *)&temp, 1);

            arr[i][0] = (double)temp;

            // std::cout << (double)temp << ' ';
            // std::cout << "\n\n";
        }
    }
}

double** ReshapeData(OPTION pOption) {
    if (pOption == TESTIMAGE) {
        double **Test_data = new double *[NUMBER_OF_TEST_DATA];
        IMAGE_Reader(TEST_IMAGE_FILE, Test_data);

        return Test_data;
    } else if (pOption == TESTLABEL) {
        double **Test_label = new double *[NUMBER_OF_TEST_DATA];
        LABEL_Reader(TEST_LABEL_FILE, Test_label);

        return Test_label;
    } else if (pOption == TRAINIMAGE) {
        double **Train_data = new double *[NUMBER_OF_TRAIN_DATA];
        IMAGE_Reader(TRAIN_IMAGE_FILE, Train_data);

        return Train_data;
    } else if (pOption == TRAINLABEL) {
        double **Train_label = new double *[NUMBER_OF_TRAIN_DATA];
        LABEL_Reader(TRAIN_LABEL_FILE, Train_label);

        return Train_label;
    } else return NULL;
}

DataSet* CreateDataSet() {
    DataSet *dataset = new DataSet();

    // double **Test_image = new double *[NUMBER_OF_TEST_DATA];
    // IMAGE_Reader(TEST_IMAGE_FILE, Test_image);

    dataset->SetTestImage(ReshapeData(TESTIMAGE));

    // double **Test_label = new double *[NUMBER_OF_TEST_DATA];
    // LABEL_Reader(TEST_LABEL_FILE, Test_label);

    dataset->SetTestLabel(ReshapeData(TESTLABEL));

    // double **Train_image = new double *[NUMBER_OF_TRAIN_DATA];
    // IMAGE_Reader(TRAIN_IMAGE_FILE, Train_image);

    dataset->SetTrainImage(ReshapeData(TRAINIMAGE));
    //
    // double **Train_label = new double *[NUMBER_OF_TRAIN_DATA];
    // LABEL_Reader(TRAIN_LABEL_FILE, Train_label);

    dataset->SetTrainLabel(ReshapeData(TRAINLABEL));

    return dataset;
}

// Tensor** CreateDataPair(OPTION pOption, int batch_size, double **origin_image, double **origin_label) {
//     // OPTION image_option   = DEFAULT;
//     // OPTION label_option   = DEFAULT;
//     int number_of_data = 0;
//     int random         = 0;
//
//     if (pOption == TEST) {
//         // image_option   = TESTIMAGE;
//         // label_option   = TESTLABEL;
//         number_of_data = NUMBER_OF_TEST_DATA;
//     } else if (pOption == TRAIN) {
//         // image_option   = TRAINIMAGE;
//         // label_option   = TRAINLABEL;
//         number_of_data = NUMBER_OF_TRAIN_DATA;
//     } else {
//         std::cout << "invalid OPTION!" << '\n';
//         exit(0);
//     }
//
//     // create input image data
//     double *****image_data = new double ****[1];
//     image_data[0] = new double ***[batch_size];
//
//     int *image_shape = new int[5] { 1, batch_size, 1, 1, DIMENSION_OF_MNIST_IMAGE };
//     int  image_rank  = 5;
//     // double **origin_image = ReshapeData(image_option);
//
//     // create input label data
//     double *****label_data = new double ****[1];
//     label_data[0] = new double ***[batch_size];
//
//     int *label_shape = new int[5] { 1, batch_size, 1, 1, 10 };
//     int  label_rank  = 5;
//     // double **origin_label = ReshapeData(label_option);
//
//     // 무작위 선택
//     srand(time(NULL));
//
//     for (int ba = 0; ba < batch_size; ba++) {
//         random = rand() % number_of_data;
//
//         image_data[0][ba]    = new double **[1];
//         image_data[0][ba][0] = new double *[1];
//         // image_data[0][ba][0][0] = origin_image[random];
//         image_data[0][ba][0][0] = new double[DIMENSION_OF_MNIST_IMAGE];
//
//         for (int dim = 0; dim < DIMENSION_OF_MNIST_IMAGE; dim++) {
//             image_data[0][ba][0][0][dim] = origin_image[random][dim];
//         }
//
//         label_data[0][ba]       = new double **[1];
//         label_data[0][ba][0]    = new double *[1];
//         label_data[0][ba][0][0] = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
//         // std::cout << (int)origin_label[random][0] << '\n';
//         label_data[0][ba][0][0][(int)origin_label[random][0]] = 1.0;
//     }
//
//     // for (int num = 0; num < number_of_data; num++) {
//     // delete[] origin_image[num];
//     // delete[] origin_label[num];
//     // }
//     // delete origin_image;
//     // delete origin_label;
//
//     Tensor *image_Tensor = new Tensor(image_data, image_shape, image_rank);
//     Tensor *label_Tensor = new Tensor(label_data, label_shape, label_rank);
//
//     Tensor **data_pair = new Tensor *[2];
//     data_pair[0] = image_Tensor;
//     data_pair[1] = label_Tensor;
//
//     return data_pair;
// }

// int main() {
// DataSet *dataset = CreateDataSet();
//
// Tensor **data_pair = CreateDataPair(TEST, 20, dataset->Test_image, dataset->Test_label);
//
//// Tensor **data_pair = CreateDataPair(TEST, 20);
//
//// data_pair[0]->PrintShape();
//// data_pair[0]->PrintData();
//// data_pair[1]->PrintShape();
//// data_pair[1]->PrintData();
//
// return 0;
// }
