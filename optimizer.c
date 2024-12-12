#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#define IMG_SIZE 28
#define VECTOR_SIZE (IMG_SIZE * IMG_SIZE + 1) // image + bias
#define TRAIN_SIZE 800
#define TEST_SIZE 200
#define NUM_CLASSES 2

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
    double w_first;
} Model;

// declarations
void load_dataset(const char *image_file, Dataset *dataset);
void gd_train_model(Dataset *dataset, Dataset *testset, Model *model, double lr, int epoch);
void sgd_train_model(Dataset *dataset, Dataset *testset, Model *model, double lr, int iteration);
void adam_train_model(Dataset *dataset, Dataset *testset, Model *model, double lr, int iteration);
double test_model(Dataset *dataset, Model *model);
void log_metrics(const char *basename, int epoch, double loss, double elapsed_time, double w, double train_success, double test_success);
void log_weights(const char *basename, const char *algorithm, double *weights, int num_weights, int epoch, double w_init);

int main()
{
    srand(time(NULL));
    // init datasets
    Dataset trainset, testset;
    trainset.count = 800;
    testset.count = 200;
    // memory allocation
    trainset.images = (double **)malloc(trainset.count * sizeof(double *));
    for (int i = 0; i < trainset.count; i++)
    {
        trainset.images[i] = (double *)malloc(VECTOR_SIZE * sizeof(double));
    }
    trainset.labels = (int *)malloc(trainset.count * sizeof(int));
    testset.images = (double **)malloc(testset.count * sizeof(double *));
    for (int i = 0; i < testset.count; i++)
    {
        testset.images[i] = (double *)malloc(VECTOR_SIZE * sizeof(double));
    }
    testset.labels = (int *)malloc(testset.count * sizeof(int));

    // load datasets from binary files
    load_dataset("train_images.bin", &trainset);
    load_dataset("test_images.bin", &testset);

    // init models
    Model gdmodel, sgdmodel, adam_model;
    gdmodel.size = VECTOR_SIZE; // pixel + bias
    gdmodel.weights = (double *)malloc(VECTOR_SIZE * sizeof(double));
    sgdmodel.size = VECTOR_SIZE; // pixel + bias
    sgdmodel.weights = (double *)malloc(VECTOR_SIZE * sizeof(double));
    adam_model.size = VECTOR_SIZE; // pixel + bias
    adam_model.weights = (double *)malloc(VECTOR_SIZE * sizeof(double));

    int w_iter;
    for (w_iter = 0; w_iter < 5; w_iter++)
    {
        switch (w_iter)
        {
        case 0:
            gdmodel.w_first = 1;
            sgdmodel.w_first = 1;
            adam_model.w_first = 1;
            break;

        case 1:
            gdmodel.w_first = 0.1;
            sgdmodel.w_first = 0.1;
            adam_model.w_first = 0.1;
            break;

        case 2:
            gdmodel.w_first = 0.01;
            sgdmodel.w_first = 0.01;
            adam_model.w_first = 0.01;
            break;

        case 3:
            gdmodel.w_first = 0.001;
            sgdmodel.w_first = 0.001;
            adam_model.w_first = 0.001;
            break;

        case 4:
            gdmodel.w_first = 0;
            sgdmodel.w_first = 0;
            adam_model.w_first = 0;
            break;
        default:
            break;
        }
        // init random weight values (zero average)
        for (int i = 0; i < gdmodel.size; i++)
        {
            int sign = -1;
            for (size_t j = 0; j < rand() % 2; j++)
            {
                sign *= sign;
            }
            double rnd = sign * ((double)rand() / RAND_MAX) * gdmodel.w_first; // Random values in [-w, w]
            // double rnd = gdmodel.w_first; // Random values in [-w, w]

            gdmodel.weights[i] = rnd;
            sgdmodel.weights[i] = rnd;
            adam_model.weights[i] = rnd;
        }

        // train model with gradient descent
        gd_train_model(&trainset, &testset, &gdmodel, 0.01, 100);
        // test_model(trainset, gdmodel);
        // test_model(testset, gdmodel);

        // train model with stochastic gradient descent
        sgd_train_model(&trainset, &testset, &sgdmodel, 0.00001, 1600);
        // test_model(trainset, sgdmodel);
        // test_model(testset, sgdmodel);

        // train model with ADAM
        adam_train_model(&trainset, &testset, &adam_model, 0.00001, 1600);
        // test_model(trainset, adam_model);
        // test_model(testset, adam_model);
    }
    // free memory
    for (int i = 0; i < trainset.count; i++)
        free(trainset.images[i]);
    free(trainset.images);
    free(trainset.labels);
    for (int i = 0; i < testset.count; i++)
        free(testset.images[i]);
    free(testset.images);
    free(testset.labels);
    free(gdmodel.weights);
    free(sgdmodel.weights);
    free(adam_model.weights);
}

void load_dataset(const char *image_file, Dataset *dataset)
{
    srand(time(NULL));

    // open binary files
    FILE *image_fp = fopen(image_file, "rb");
    if (!image_fp)
    {
        fprintf(stderr, "Error: Could not open dataset file.\n");
        exit(EXIT_FAILURE);
    }
    // firstly get all data to 1d array
    uint8_t *raw_images = (uint8_t *)malloc(dataset->count * IMG_SIZE * IMG_SIZE * sizeof(uint8_t));
    fread(raw_images, sizeof(uint8_t), dataset->count * IMG_SIZE * IMG_SIZE, image_fp);

    fclose(image_fp);

    // Normalize images and manually assign labels (-1 or 1)
    for (int i = 0; i < dataset->count; i++)
    {
        for (int j = 0; j < IMG_SIZE * IMG_SIZE; j++)
        {
            dataset->images[i][j] = raw_images[i * IMG_SIZE * IMG_SIZE + j] / 255.0;
        }
        dataset->images[i][IMG_SIZE * IMG_SIZE] = 1.0;          // Add bias term
        dataset->labels[i] = (i < dataset->count / 2) ? -1 : 1; // Binary labels
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

void gd_train_model(Dataset *dataset, Dataset *testset, Model *model, double lr, int epoch)
{
    // w_new = w_old - lr * dß/dw

    // memory allocation
    double *grad = (double *)malloc(VECTOR_SIZE * sizeof(double));
    double *wx = (double *)malloc(dataset->count * sizeof(double));
    double tanh_wx, loss, wxx;

    loss = 0;
    for (int j = 0; j < dataset->count; j++)
    {
        wxx = 0; // wx vector calculation
        for (int k = 0; k < model->size; k++)
        {
            wxx += model->weights[k] * dataset->images[k][j];
        }
        tanh_wx = tanh(wxx);
        loss += ((dataset->labels[j] - tanh_wx) * (dataset->labels[j] - tanh_wx));
    } // calculate loss
    loss /= dataset->count;

    double train_success = test_model(dataset, model);
    double test_success = test_model(testset, model);
    log_metrics("gd_metrics", 0, loss, 0, model->w_first, train_success, test_success);
    log_weights("weights_log", "gd", model->weights, model->size, 0, model->w_first);

    clock_t start = clock();

    // main loop
    for (int i = 0; i < epoch; i++)
    {
        // compute W * X vector
        for (int j = 0; j < dataset->count; j++)
        {
            wx[j] = 0;
            for (int k = 0; k < model->size; k++)
            {
                wx[j] += model->weights[k] * dataset->images[j][k];
            }
        }

        // update weights
        for (int j = 0; j < model->size; j++)
        {
            // 1 * pixel+1  **  pixel+1 * samplecount
            // compute grad(dß/dw) vector
            grad[j] = 0;
            for (int k = 0; k < dataset->count; k++)
            {

                tanh_wx = tanh(wx[k]);
                grad[j] += -2 * (dataset->labels[k] - tanh_wx) * (1 - tanh_wx * tanh_wx) * dataset->images[k][j];
            }
            model->weights[j] -= lr * grad[j];
        }

        // compute loss
        loss = 0;
        for (int j = 0; j < dataset->count; j++)
        {
            tanh_wx = tanh(wx[j]);
            loss += ((dataset->labels[j] - tanh_wx) * (dataset->labels[j] - tanh_wx));
        }
        clock_t end = clock();
        double elapsed = (end - start) / 1000.0;

        loss /= dataset->count;
        // printf("epoch: %d\t loss: %lf\t time: %lf\n", i, loss, elapsed);
        train_success = test_model(dataset, model);
        test_success = test_model(testset, model);
        log_metrics("gd_metrics", i + 1, loss, elapsed, model->w_first, train_success, test_success);
        log_weights("weights_log", "gd", model->weights, model->size, i + 1, model->w_first);

        // printf("%d\t \n", dataset->labels[i]);
    }
    free(grad);
}

void sgd_train_model(Dataset *dataset, Dataset *testset, Model *model, double lr, int iteration)
{
    // w_new = w_old - lr * J_transpose * r

    // memory allocation
    double *grad = (double *)malloc(VECTOR_SIZE * sizeof(double));
    double wx, tanh_wx, loss, elapsed;
    int ii;

    loss = 0;
    for (int j = 0; j < dataset->count; j++)
    {
        for (int k = 0; k < model->size; k++)
        {
            wx = 0;

            wx += model->weights[k] * dataset->images[k][j];
        }
        tanh_wx = tanh(wx);
        loss += ((dataset->labels[j] - tanh_wx) * (dataset->labels[j] - tanh_wx));
    }
    loss /= dataset->count;

    double train_success = test_model(dataset, model);
    double test_success = test_model(testset, model);
    log_metrics("sgd_metrics", 0, loss, 0, model->w_first, train_success, test_success);
    log_weights("weights_log", "sgd", model->weights, model->size, 0, model->w_first);

    clock_t start = clock();

    // main loop change sample each iteration
    for (int i = 0; i < iteration; i++)
    {
        // w_old = w_new - lr * J[i]*r

        // 1 * pixel+1  **  pixel+1 * samplecount
        ii = i % dataset->count; // inside iteration

        // compute wx vector
        wx = 0;
        for (int j = 0; j < model->size; j++)
        {
            wx += model->weights[j] * dataset->images[ii][j];
        }

        grad[ii] = 0; // gradient init
        tanh_wx = tanh(wx);

        for (int j = 0; j < dataset->count; j++)
        { // compute gradient
            grad[ii] += -2 * (dataset->labels[j] - tanh_wx) * (1 - tanh_wx * tanh_wx) * dataset->images[j][ii];
        }
        model->weights[ii] -= lr * grad[ii];
        loss = 0;

        // calculate loss
        for (int i = 0; i < dataset->count; i++)
        {
            double wx = 0.0;
            for (int j = 0; j < model->size; j++)
            {
                wx += model->weights[j] * dataset->images[i][j];
            }
            double tanh_wxi = tanh(wx);
            loss += ((dataset->labels[i] - tanh_wxi) * (dataset->labels[i] - tanh_wxi));
        }
        clock_t end = clock();
        elapsed = (end - start) / 1000.0; // timer
        loss /= dataset->count;

        // printf("iteration: %d\tloss: %lf\ttime: %lf\n", i, loss, elapsed);
        train_success = test_model(dataset, model);
        test_success = test_model(testset, model);
        log_metrics("sgd_metrics", i + 1, loss, elapsed, model->w_first, train_success, test_success);
        log_weights("weights_log", "sgd", model->weights, model->size, i + 1, model->w_first);
    }

    free(grad);
}

void adam_train_model(Dataset *dataset, Dataset *testset, Model *model, double lr, int iteration)
{
    double alpha = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 0.00000001;
    double *m = (double *)malloc(model->size * sizeof(double));
    double *v = (double *)malloc(model->size * sizeof(double));
    double *g = (double *)malloc(model->size * sizeof(double));
    double mhat, vhat, elapsed;
    double wx, tanh_wx, loss;
    int ii;

    loss = 0;
    for (int j = 0; j < dataset->count; j++)
    {
        wx = 0; // init wx

        for (int k = 0; k < model->size; k++)
        { // calculate w.x
            wx += model->weights[k] * dataset->images[k][j];
        }
        tanh_wx = tanh(wx);
        loss += ((dataset->labels[j] - tanh_wx) * (dataset->labels[j] - tanh_wx));
    } // calculate loss
    loss /= dataset->count;

    double train_success = test_model(dataset, model);
    double test_success = test_model(testset, model);
    log_metrics("adam_metrics", 0, loss, 0, model->w_first, train_success, test_success);
    log_weights("weights_log", "adam", model->weights, model->size, 0, model->w_first);

    // init zero
    for (int i = 0; i < model->size; i++)
    {
        m[i] = 0;
        v[i] = 0;
    }

    clock_t start = clock(); // start timing

    // main loop
    for (int t = 0; t < iteration; t++)
    {
        wx = 0;
        loss = 0;
        ii = t % dataset->count; // inside iteration
        for (int j = 0; j < model->size; j++)
        { // calculate w.x
            wx += model->weights[j] * dataset->images[ii][j];
        }

        tanh_wx = tanh(wx);

        // stochastic gradient sample
        for (int j = 0; j < model->size; j++)
        {
            g[j] = -2 * (dataset->labels[ii] - tanh_wx) * (1 - tanh_wx * tanh_wx) * dataset->images[ii][j];
        }

        // update m and v
        for (int i = 0; i < model->size; i++)
        {
            m[i] = beta1 * m[i] + (1 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
        }

        // bias correction
        for (int j = 0; j < model->size; j++)
        {
            mhat = m[j] / (1 - pow(beta1, t + 1)); // Corrected first moment
            vhat = v[j] / (1 - pow(beta2, t + 1)); // Corrected second moment

            // Update weights
            model->weights[j] -= alpha * mhat / (sqrt(vhat) + eps);
        }
        clock_t end = clock();
        elapsed = ((double)(end - start)) / CLOCKS_PER_SEC; // Calculate elapsed time

        // loss
        loss = 0;
        for (int i = 0; i < dataset->count; i++)
        {
            double wx = 0.0;
            for (int j = 0; j < model->size; j++)
            {
                wx += model->weights[j] * dataset->images[i][j];
            }
            double tanh_wxi = tanh(wx);
            loss += ((dataset->labels[i] - tanh_wxi) * (dataset->labels[i] - tanh_wxi));
        }
        loss /= dataset->count; //calculate loss
        // printf("iteration: %d\t loss: %lf\t time: %lf seconds\n", t, loss, elapsed);
        train_success = test_model(dataset, model);
        test_success = test_model(testset, model);
        log_metrics("adam_metrics", t + 1, loss, elapsed, model->w_first, train_success, test_success);
        log_weights("weights_log", "adam", model->weights, model->size, t + 1, model->w_first);
    }
}

double test_model(Dataset *dataset, Model *model)
{
    int correct = 0;
    // double loss = 0.0;
    /**/
    for (int i = 0; i < dataset->count; i++)
    {
        double wx = 0.0;
        for (int j = 0; j < model->size; j++)
        { // calculate w.x
            wx += model->weights[j] * dataset->images[i][j];
        }
        double tanh_wxi = tanh(wx);

        //if tanh(wx) > 0 prediction is tshirt
        int prediction = tanh_wxi > 0 ? 1 : -1;

        if (dataset->labels[i] == prediction)
        {
            correct++;
        }
        // loss += ((dataset->labels[i] - tanh_wxi) * (dataset->labels[i] - tanh_wxi));
    }
    // loss /= dataset->count;*/
    double acc = 100 * correct / dataset->count;

    // printf("Accuracy: %.2f%%\tloss: %lf\n", acc, loss);

    return acc;
}

void log_metrics(const char *basename, int epoch, double loss, double elapsed_time, double w, double train_success, double test_success)
{
    char filename[256];
    sprintf(filename, "%s_w_%.4f.csv", basename, w);
    FILE *fp = fopen(filename, "a"); // "a" ile açmak, dosyaya ekleme yapar
    if (fp == NULL)
    {
        fprintf(stderr, "Dosya açılamadı: %s\n", filename);
        return;
    }

    fprintf(fp, "%d, %lf, %lf, %lf, %lf\n", epoch, loss, elapsed_time, train_success, test_success); // epoch, loss, time yaz
    fclose(fp);
}

void log_weights(const char *basename, const char *algorithm, double *weights, int num_weights, int epoch, double w_init)
{
    char filename[256];
    sprintf(filename, "%s_%s_w_%.4f.csv", basename, algorithm, w_init); // Algoritma ve w değeri içeren dosya adı
    FILE *fp = fopen(filename, "a");                                    // "a" ile açmak, dosyaya ekleme yapar
    if (fp == NULL)
    {
        fprintf(stderr, "Dosya açılamadı: %s\n", filename);
        return;
    }

    fprintf(fp, "%d", epoch); // Epoch numarasını yaz
    for (int i = 0; i < num_weights; i++)
    {
        fprintf(fp, ",%lf", weights[i]); // Her ağırlığı yaz
    }
    fprintf(fp, "\n");
    fclose(fp);
}
