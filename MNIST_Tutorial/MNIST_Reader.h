#include <fstream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>

#include "..//Header//Tensor.h"

#define DIMENSION_OF_MNIST_IMAGE    784

#define NUMBER_OF_TEST_DATA         10000
#define NUMBER_OF_TRAIN_DATA        60000

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

int random_generator(int i) {
    return rand() % i;
}

template<typename DTYPE>
class MNISTDataSet {
private:
    // 직접 해제
    DTYPE **Test_image = NULL;
    DTYPE **Test_label = NULL;
    DTYPE **Train_image = NULL;
    DTYPE **Train_label = NULL;

    // 따로 해제
    Tensor<DTYPE> *Test_image_feed = NULL;
    Tensor<DTYPE> *Test_label_feed = NULL;
    Tensor<DTYPE> *Train_image_feed = NULL;
    Tensor<DTYPE> *Train_label_feed = NULL;

    Tensor<DTYPE> *Test_Data_pair[2] = { NULL, NULL };
    Tensor<DTYPE> *Train_Data_pair[2] = { NULL, NULL };

    vector<int> *shuffled_list_for_test = NULL;
    vector<int> *shuffled_list_for_train = NULL;

    int Recallnum_of_test = 0;
    int Recallnum_of_train = 0;

public:
    MNISTDataSet() {
        Alloc();
    }

    virtual ~MNISTDataSet() {
        Delete();
    }

    void Alloc() {
        int Number_of_test[NUMBER_OF_TEST_DATA] = { 0 };

        for (int i = 0; i < NUMBER_OF_TEST_DATA; i++) {
            Number_of_test[i] = i;
        }

        int Number_of_train[NUMBER_OF_TRAIN_DATA] = { 0 };

        for (int i = 0; i < NUMBER_OF_TRAIN_DATA; i++) {
            Number_of_train[i] = i;
        }

        shuffled_list_for_test = new vector<int>(Number_of_test, Number_of_test + NUMBER_OF_TEST_DATA);

        shuffled_list_for_train = new vector<int>(Number_of_train, Number_of_train + NUMBER_OF_TRAIN_DATA);
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

    // After we see test data at one time
    void CreateTestDataPair(int batch_size) {
        // if (((batch_size * Recallnum_of_test) % NUMBER_OF_TEST_DATA) == 0) ShuffleDataPair(TEST, batch_size);

        CreateTestDataPair_STEP2(batch_size);

        Recallnum_of_test++;
    }

    void CreateTrainDataPair(int batch_size) {
        if ((batch_size * Recallnum_of_train % NUMBER_OF_TRAIN_DATA) == 0) ShuffleDataPair(TRAIN, batch_size);

        CreateTrainDataPair_STEP2(batch_size);

        Recallnum_of_train++;
    }

    void ShuffleDataPair(OPTION pOption, int batch_size) {
        // std::cout << "shuffle" << '\n';
        srand(unsigned(time(0)));
        // int  number_of_data = 0;
        vector<int> *shuffled_list = NULL;

        if (pOption == TEST) {
            // number_of_data = NUMBER_OF_TEST_DATA;
            shuffled_list = shuffled_list_for_test;
        } else if (pOption == TRAIN) {
            // number_of_data = NUMBER_OF_TRAIN_DATA;
            shuffled_list = shuffled_list_for_train;
        } else {
            cout << "invalid OPTION!" << '\n';
            exit(0);
        }

        random_shuffle(shuffled_list->begin(), shuffled_list->end());
        random_shuffle(shuffled_list->begin(), shuffled_list->end(), random_generator);

        // for (vector<int>::iterator it = shuffled_list->begin(); it != shuffled_list->end(); ++it) cout << ' ' << *it;
    }

    void CreateTestDataPair_STEP2(int batch_size) {
        int number_of_data = NUMBER_OF_TEST_DATA;
        int Recallnum = Recallnum_of_test;
        int start_point = 0;
        int cur_point = 0;
        DTYPE **origin_image = Test_image;
        DTYPE **origin_label = Test_label;

        vector<int> *shuffled_list = shuffled_list_for_test;

        // create input image data
        DTYPE *****image_data = new DTYPE * ** *[1];

        image_data[0] = new DTYPE * * *[batch_size];

        int *image_shape = new int[5] { 1, batch_size, 1, 1, DIMENSION_OF_MNIST_IMAGE };
        int  image_rank = 5;
        // double **origin_image = ReshapeData(image_option);

        // create input label data
        DTYPE *****label_data = new DTYPE * ** *[1];
        label_data[0] = new DTYPE * * *[batch_size];

        int *label_shape = new int[5] { 1, batch_size, 1, 1, 10 };
        int  label_rank = 5;

        start_point = (Recallnum * batch_size) % number_of_data;

        for (int ba = 0; ba < batch_size; ba++) {
            cur_point = (*shuffled_list)[start_point + ba];

            // cout << cur_point << ' ';

            image_data[0][ba] = new DTYPE * *[1];
            image_data[0][ba][0] = new DTYPE *[1];
            // image_data[0][ba][0][0] = origin_image[random];
            image_data[0][ba][0][0] = new DTYPE[DIMENSION_OF_MNIST_IMAGE];

            for (int dim = 0; dim < DIMENSION_OF_MNIST_IMAGE; dim++) {
                image_data[0][ba][0][0][dim] = origin_image[cur_point][dim];
            }

            // ---------------------------------------------------------------------

            label_data[0][ba] = new DTYPE * *[1];
            label_data[0][ba][0] = new DTYPE *[1];
            label_data[0][ba][0][0] = new DTYPE[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            // cout << (int)origin_label[random][0] << '\n';
            label_data[0][ba][0][0][(int)origin_label[cur_point][0]] = 1.0;

            // cout << random << ' ';
        }
        // cout << '\n';

        Test_image_feed = new Tensor<DTYPE>(image_data, image_shape, image_rank);

        Test_label_feed = new Tensor<DTYPE>(label_data, label_shape, label_rank);
    }

    void CreateTrainDataPair_STEP2(int batch_size) {
        int number_of_data = NUMBER_OF_TRAIN_DATA;
        int Recallnum = Recallnum_of_train;
        int start_point = 0;
        int cur_point = 0;
        DTYPE **origin_image = Train_image;
        DTYPE **origin_label = Train_label;

        vector<int> *shuffled_list = shuffled_list_for_train;


        // create input image data
        DTYPE *****image_data = new DTYPE * ** *[1];

        image_data[0] = new DTYPE * * *[batch_size];

        int *image_shape = new int[5] { 1, batch_size, 1, 1, DIMENSION_OF_MNIST_IMAGE };
        int  image_rank = 5;
        // double **origin_image = ReshapeData(image_option);

        // create input label data
        DTYPE *****label_data = new DTYPE * ** *[1];
        label_data[0] = new DTYPE * * *[batch_size];

        int *label_shape = new int[5] { 1, batch_size, 1, 1, 10 };
        int  label_rank = 5;

        start_point = (Recallnum * batch_size) % number_of_data;

        for (int ba = 0; ba < batch_size; ba++) {
            cur_point = (*shuffled_list)[start_point + ba];

            // cout << cur_point << ' ';

            image_data[0][ba] = new DTYPE * *[1];
            image_data[0][ba][0] = new DTYPE *[1];
            // image_data[0][ba][0][0] = origin_image[random];
            image_data[0][ba][0][0] = new DTYPE[DIMENSION_OF_MNIST_IMAGE];

            for (int dim = 0; dim < DIMENSION_OF_MNIST_IMAGE; dim++) {
                image_data[0][ba][0][0][dim] = origin_image[cur_point][dim];
            }

            // ---------------------------------------------------------------------

            label_data[0][ba] = new DTYPE * *[1];
            label_data[0][ba][0] = new DTYPE *[1];
            label_data[0][ba][0][0] = new DTYPE[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            // cout << (int)origin_label[random][0] << '\n';
            label_data[0][ba][0][0][(int)origin_label[cur_point][0]] = 1.0;

            // cout << random << ' ';
        }
        // cout << '\n';

        Train_image_feed = new Tensor<DTYPE>(image_data, image_shape, image_rank);
        Train_label_feed = new Tensor<DTYPE>(label_data, label_shape, label_rank);

    }

    void SetTestImage(DTYPE **pTest_image) {
        Test_image = pTest_image;
    }

    void SetTestLabel(DTYPE **pTest_label) {
        Test_label = pTest_label;
    }

    void SetTrainImage(DTYPE **pTrain_image) {
        Train_image = pTrain_image;
    }

    void SetTrainLabel(DTYPE **pTrain_label) {
        Train_label = pTrain_label;
    }

    Tensor<DTYPE>* GetTestFeedImage() {
        return Test_image_feed;
    }

    Tensor<DTYPE>* GetTrainFeedImage() {
        return Train_image_feed;
    }

    Tensor<DTYPE>* GetTestFeedLabel() {
        return Test_label_feed;
    }

    Tensor<DTYPE>* GetTrainFeedLabel() {
        return Train_label_feed;
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

template<typename DTYPE>
void IMAGE_Reader(string DATAPATH, DTYPE **arr) {
    // arr.resize(NumberOfImages, vector<double>(DataOfAnImage));
    ifstream fin;

    fin.open(DATAPATH, ios::binary);

    if (fin.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        fin.read((char *)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);

        fin.read((char *)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        fin.read((char *)&n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);

        fin.read((char *)&n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        int dim_of_image = n_rows * n_cols;

        for (int i = 0; i < number_of_images; ++i) {
            arr[i] = new DTYPE[dim_of_image];

            for (int d = 0; d < dim_of_image; ++d) {
                unsigned char temp = 0;
                fin.read((char *)&temp, sizeof(temp));
                arr[i][d] = (DTYPE)temp / 255.0;
                // cout << arr[i][d] << ' ';
            }
            // cout << "\n\n";
        }
    }
}

template<typename DTYPE>
void LABEL_Reader(string DATAPATH, DTYPE **arr) {
    // arr.resize(NumberOfImages, vector<double>(DataOfAnImage));
    ifstream fin;

    fin.open(DATAPATH, ios::binary);

    if (fin.is_open()) {
        int magic_number = 0;
        int number_of_labels = 0;

        fin.read((char *)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);

        fin.read((char *)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = ReverseInt(number_of_labels);

        // cout << number_of_labels << '\n';

        for (int i = 0; i < number_of_labels; ++i) {
            arr[i] = new DTYPE[1];

            unsigned char temp = 0;
            fin.read((char *)&temp, 1);

            arr[i][0] = (DTYPE)temp;

            // cout << (double)temp << ' ';
            // cout << "\n\n";
        }
    }
}

template<typename DTYPE>
DTYPE** ReshapeData(OPTION pOption) {
    if (pOption == TESTIMAGE) {
        DTYPE **Test_data = new DTYPE *[NUMBER_OF_TEST_DATA];
        IMAGE_Reader(TEST_IMAGE_FILE, Test_data);

        return Test_data;
    } else if (pOption == TESTLABEL) {
        DTYPE **Test_label = new DTYPE *[NUMBER_OF_TEST_DATA];
        LABEL_Reader(TEST_LABEL_FILE, Test_label);

        return Test_label;
    } else if (pOption == TRAINIMAGE) {
        DTYPE **Train_data = new DTYPE *[NUMBER_OF_TRAIN_DATA];
        IMAGE_Reader(TRAIN_IMAGE_FILE, Train_data);

        return Train_data;
    } else if (pOption == TRAINLABEL) {
        DTYPE **Train_label = new DTYPE *[NUMBER_OF_TRAIN_DATA];
        LABEL_Reader(TRAIN_LABEL_FILE, Train_label);

        return Train_label;
    } else return NULL;
}

template<typename DTYPE>
MNISTDataSet<DTYPE>* CreateMNISTDataSet() {
    MNISTDataSet<DTYPE> *dataset = new MNISTDataSet<DTYPE>();

    dataset->SetTestImage(ReshapeData<DTYPE>(TESTIMAGE));

    dataset->SetTestLabel(ReshapeData<DTYPE>(TESTLABEL));

    dataset->SetTrainImage(ReshapeData<DTYPE>(TRAINIMAGE));

    dataset->SetTrainLabel(ReshapeData<DTYPE>(TRAINLABEL));

    return dataset;
}
