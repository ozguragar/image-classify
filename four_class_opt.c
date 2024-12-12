#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#define IMG_SIZE 28
#define VECTOR_SIZE (IMG_SIZE * IMG_SIZE + 1) // Flattened image + bias
#define TRAIN_SIZE 1500
#define TEST_SIZE 1500
#define NUM_CLASSES 4

typedef struct
{
    double **images; // (samples x (pixels + 1))
    int *labels;     // samples
    int count;       // sample count
} Dataset;

typedef struct
{
    double *weights; // pixels
    int size;        // pixels + 1
} Model;

void load_dataset(const char *image_file, const char *label_file, Dataset *dataset, int classcode);
void gd_train_model(Dataset *dataset, Model *model, double lr, int epoch);
void test_model(Dataset dataset, Model model);
void test_model_all(Dataset dataset, Model model0, Model model1, Model model2, Model model3);

int main()
{
    srand(time(NULL));
    // init datasets
    Dataset trainset0, trainset1, trainset2, trainset3, testset0, testset1, testset2, testset3;
    trainset0.count = 1500;
    testset0.count = 1500;
    trainset0.images = (double **)malloc(trainset0.count * sizeof(double *));
    for (int i = 0; i < trainset0.count; i++)
    {
        trainset0.images[i] = (double *)malloc(VECTOR_SIZE * sizeof(double));
    }
    trainset0.labels = (int *)malloc(trainset0.count * sizeof(int));
    testset0.images = (double **)malloc(testset0.count * sizeof(double *));
    for (int i = 0; i < testset0.count; i++)
    {
        testset0.images[i] = (double *)malloc(VECTOR_SIZE * sizeof(double));
    }
    testset0.labels = (int *)malloc(testset0.count * sizeof(int));

    load_dataset("train_images_ts.bin", "train_labels_ts.bin", &trainset0, 0);
    load_dataset("test_images_ts.bin", "test_labels_ts.bin", &testset0, 0);

    trainset1.count = 1500;
    testset1.count = 1500;
    trainset1.images = (double **)malloc(trainset1.count * sizeof(double *));
    for (int i = 0; i < trainset1.count; i++)
    {
        trainset1.images[i] = (double *)malloc(VECTOR_SIZE * sizeof(double));
    }
    trainset1.labels = (int *)malloc(trainset1.count * sizeof(int));
    testset1.images = (double **)malloc(testset1.count * sizeof(double *));
    for (int i = 0; i < testset1.count; i++)
    {
        testset1.images[i] = (double *)malloc(VECTOR_SIZE * sizeof(double));
    }
    testset1.labels = (int *)malloc(testset1.count * sizeof(int));

    load_dataset("train_images_ab.bin", "train_labels_ab.bin", &trainset1, 1);
    load_dataset("test_images_ab.bin", "test_labels_ab.bin", &testset1, 1);

    trainset2.count = 1500;
    testset2.count = 1500;
    trainset2.images = (double **)malloc(trainset2.count * sizeof(double *));
    for (int i = 0; i < trainset2.count; i++)
    {
        trainset2.images[i] = (double *)malloc(VECTOR_SIZE * sizeof(double));
    }
    trainset2.labels = (int *)malloc(trainset2.count * sizeof(int));
    testset2.images = (double **)malloc(testset2.count * sizeof(double *));
    for (int i = 0; i < testset2.count; i++)
    {
        testset2.images[i] = (double *)malloc(VECTOR_SIZE * sizeof(double));
    }
    testset2.labels = (int *)malloc(testset2.count * sizeof(int));

    load_dataset("train_images_bag.bin", "train_labels_bag.bin", &trainset2, 2);
    load_dataset("test_images_bag.bin", "test_labels_bag.bin", &testset2, 2);

    trainset3.count = 1500;
    testset3.count = 1500;
    trainset3.images = (double **)malloc(trainset3.count * sizeof(double *));
    for (int i = 0; i < trainset3.count; i++)
    {
        trainset3.images[i] = (double *)malloc(VECTOR_SIZE * sizeof(double));
    }
    trainset3.labels = (int *)malloc(trainset3.count * sizeof(int));
    testset3.images = (double **)malloc(testset3.count * sizeof(double *));
    for (int i = 0; i < testset3.count; i++)
    {
        testset3.images[i] = (double *)malloc(VECTOR_SIZE * sizeof(double));
    }
    testset3.labels = (int *)malloc(testset3.count * sizeof(int));

    load_dataset("train_images_tr.bin", "train_labels_tr.bin", &trainset3, 3);
    load_dataset("test_images_tr.bin", "test_labels_tr.bin", &testset3, 3);

    Dataset testsetall;
    testsetall.count = 2000;
    testsetall.images = (double **)malloc(testsetall.count * sizeof(double *));
    for (int i = 0; i < testsetall.count; i++)
    {
        testsetall.images[i] = (double *)malloc(VECTOR_SIZE * sizeof(double));
    }
    testsetall.labels = (int *)malloc(testsetall.count * sizeof(int));

    load_dataset("test_images_all.bin", "test_labels_all.bin", &testsetall, 0);

    Model model0, model1, model2, model3;
    model0.size = VECTOR_SIZE;
    model0.weights = (double *)malloc(VECTOR_SIZE * sizeof(double));
    for (int i = 0; i < model0.size; i++)
    {
        int sign = -1;
        for (size_t j = 0; j < rand() % 2; j++)
        {
            sign *= sign;
        }
        model0.weights[i] = sign * ((double)rand() / RAND_MAX) * 0.01; // Random values in [0, 0.01]
    }
    model1.size = VECTOR_SIZE;
    model1.weights = (double *)malloc(VECTOR_SIZE * sizeof(double));
    for (int i = 0; i < model1.size; i++)
    {
        int sign = -1;
        for (size_t j = 0; j < rand() % 2; j++)
        {
            sign *= sign;
        }
        model1.weights[i] = sign * ((double)rand() / RAND_MAX) * 0.01; // Random values in [0, 0.01]
    }
    model2.size = VECTOR_SIZE;
    model2.weights = (double *)malloc(VECTOR_SIZE * sizeof(double));
    for (int i = 0; i < model2.size; i++)
    {
        int sign = -1;
        for (size_t j = 0; j < rand() % 2; j++)
        {
            sign *= sign;
        }
        model2.weights[i] = sign * ((double)rand() / RAND_MAX) * 0.01; // Random values in [0, 0.01]
    }
    model3.size = VECTOR_SIZE;
    model3.weights = (double *)malloc(VECTOR_SIZE * sizeof(double));
    for (int i = 0; i < model3.size; i++)
    {
        int sign = -1;
        for (size_t j = 0; j < rand() % 2; j++)
        {
            sign *= sign;
        }
        model3.weights[i] = sign * ((double)rand() / RAND_MAX) * 0.01; // Random values in [0, 0.01]
    }

    gd_train_model(&trainset0, &model0, 0.001, 20);
    gd_train_model(&trainset1, &model1, 0.001, 20);
    gd_train_model(&trainset2, &model2, 0.001, 20);
    gd_train_model(&trainset3, &model3, 0.001, 20);
    test_model(trainset0, model0);
    test_model(testset0, model0);
    test_model(trainset1, model1);
    test_model(testset1, model1);
    test_model(trainset2, model2);
    test_model(testset2, model2);
    test_model(trainset3, model3);
    test_model(testset3, model3);

    load_dataset("test_images_all.bin", "train_labels.bin", &testsetall, 9);
    test_model_all(testsetall, model0,model1,model2,model3);
}

void load_dataset(const char *image_file, const char *label_file, Dataset *dataset, int class_code)
{
    srand(time(NULL));

    FILE *image_fp = fopen(image_file, "rb");
    FILE *label_fp = fopen(label_file, "rb");
    if (!image_fp || !label_fp)
    {
        fprintf(stderr, "Error: Could not open dataset files.\n");
        exit(EXIT_FAILURE);
    }
    // firstly get all data to 1d array
    uint8_t *raw_images = (uint8_t *)malloc(dataset->count * IMG_SIZE * IMG_SIZE * sizeof(uint8_t));
    fread(raw_images, sizeof(uint8_t), dataset->count * IMG_SIZE * IMG_SIZE, image_fp);
    // fread(dataset->labels, sizeof(int), dataset->count, label_fp);

    fclose(image_fp);
    fclose(label_fp);

    int quat = dataset->count / 4;

    // Normalize images and assign labels (-1 or 1)
    for (int i = 0; i < dataset->count; i++)
    {
        for (int j = 0; j < IMG_SIZE * IMG_SIZE; j++)
        {
            dataset->images[i][j] = raw_images[i * IMG_SIZE * IMG_SIZE + j] / 255.0;
        }
        dataset->images[i][IMG_SIZE * IMG_SIZE] = 1.0;          // Add bias term
        dataset->labels[i] = (i < dataset->count / 2) ? -1 : 1; // Binary labels
    }
    if (class_code == 9)
    {
        {
            for (size_t i = 0; i < quat; i++)
            {
                dataset->labels[i] = 0;
            }
        }

        {
            for (size_t i = quat; i < 2 * quat; i++)
            {
                dataset->labels[i] = 1;
            }
        }

        {
            for (size_t i = 2 * quat; i < 3 * quat; i++)
            {
                dataset->labels[i] = 2;
            }
        }

        {
            for (size_t i = 3 * quat; i < 4 * quat; i++)
            {
                dataset->labels[i] = 3;
            }
        }
    }

    // Shuffle dataset
    for (int i = dataset->count - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        double *temp_image = dataset->images[i];
        dataset->images[i] = dataset->images[j];
        dataset->images[j] = temp_image;

        int temp_label = dataset->labels[i];
        dataset->labels[i] = dataset->labels[j];
        dataset->labels[j] = temp_label;
    }

    free(raw_images);
}

void gd_train_model(Dataset *dataset, Model *model, double lr, int epoch)
{
    // w_new = w_old - lr * J_transpose * r
    double *grad = (double *)malloc(VECTOR_SIZE * sizeof(double));
    double *wx = (double *)malloc(dataset->count * sizeof(double));
    double tanh_wx, loss;
    for (int i = 0; i < epoch; i++)
    {
        for (int j = 0; j < dataset->count; j++)
        {
            wx[j] = 0;
            for (int k = 0; k < model->size; k++)
            {
                wx[j] += model->weights[k] * dataset->images[j][k];
            }
        }
        for (int j = 0; j < model->size; j++)
        {
            // 1 * pixel+1  **  pixel+1 * samplecount
            grad[j] = 0;
            for (int k = 0; k < dataset->count; k++)
            {

                tanh_wx = tanh(wx[k]);
                grad[j] += -2 * (dataset->labels[k] - tanh_wx) * (1 - tanh_wx * tanh_wx) * dataset->images[k][j];
            }
            model->weights[j] -= lr * grad[j];
        }
        loss = 0;
        for (int j = 0; j < dataset->count; j++)
        {
            tanh_wx = tanh(wx[j]);
            loss += ((dataset->labels[j] - tanh_wx) * (dataset->labels[j] - tanh_wx));
        }
        loss /= dataset->count;
        printf("epoch: %d\t loss: %lf\n", i, loss);
        // printf("%d\t \n", dataset->labels[i]);
    }
    free(grad);
}

void test_model(Dataset dataset, Model model)
{
    int correct = 0;
    double loss = 0.0;

    for (int i = 0; i < dataset.count; i++)
    {
        double wx = 0.0;
        for (int j = 0; j < model.size; j++)
        {
            wx += model.weights[j] * dataset.images[i][j];
        }
        double tanh_wxi = tanh(wx);

        int prediction = tanh_wxi > 0 ? 1 : -1;

        if (dataset.labels[i] == prediction)
        {
            correct++;
        }
        loss += ((dataset.labels[i] - tanh_wxi) * (dataset.labels[i] - tanh_wxi));

        // printf("%d %d %lf\n", dataset.labels[i], prediction, tanh_wxi);
    }
    loss /= dataset.count;

    printf("Accuracy: %.2f%%\tloss: %lf\n", 100.0 * correct / dataset.count, loss);
}

void test_model_all(Dataset dataset, Model model0, Model model1, Model model2, Model model3)
{
    int correct = 0;
    double prediction0, prediction1, prediction2, prediction3 = 0;
    double tanh_wxi, loss = 0.0;

    for (int i = 0; i < dataset.count; i++)
    {
        double wx = 0.0;
        for (int j = 0; j < model0.size; j++)
        {
            wx += model0.weights[j] * dataset.images[i][j];
        }
        tanh_wxi = tanh(wx);
        

        prediction0 = tanh_wxi;
        loss += ((dataset.labels[i] - tanh_wxi) * (dataset.labels[i] - tanh_wxi));

        wx = 0.0;
        for (int j = 0; j < model1.size; j++)
        {
            wx += model1.weights[j] * dataset.images[i][j];
        }
        tanh_wxi = tanh(wx);

        prediction1 = tanh_wxi;
        loss += ((dataset.labels[i] - tanh_wxi) * (dataset.labels[i] - tanh_wxi));

        wx = 0.0;
        for (int j = 0; j < model2.size; j++)
        {
            wx += model2.weights[j] * dataset.images[i][j];
        }
        tanh_wxi = tanh(wx);

        prediction2 = tanh_wxi;
        loss += ((dataset.labels[i] - tanh_wxi) * (dataset.labels[i] - tanh_wxi));

         wx = 0.0;
        for (int j = 0; j < model3.size; j++)
        {
            wx += model3.weights[j] * dataset.images[i][j];
        }
         tanh_wxi = tanh(wx);

        prediction3 = tanh_wxi;
        loss += ((dataset.labels[i] - tanh_wxi) * (dataset.labels[i] - tanh_wxi));

        int prediction, most;
        if (prediction0 < prediction1)
        {
            prediction = 0;
            most = prediction0;
        }
        else
        {
            prediction = 1;
            most = prediction1;
        }
        if (prediction2 < most)
        {
            prediction = 2;
            most = prediction2;
        }
        if (prediction3 < most)
        {
            prediction = 3;
        }

        if (dataset.labels[i] == prediction)
        {
            correct++;
        }

        // printf("%d %d %lf\n", dataset.labels[i], prediction, tanh_wxi);
    }
    printf("Full Testing Accuracy: %.2f%%\t", 100.0 * correct / dataset.count);
}