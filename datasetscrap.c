#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define IMG_SIZE (IMG_WIDTH * IMG_HEIGHT)
#define NUM_CLASSES 2
#define NUM_IMAGES 1000
#define TRAIN_RATIO 0.8

typedef struct {
    uint8_t *images;
    uint8_t *labels;
    size_t count;
} Dataset;

// Function to load images from Fashion-MNIST
void load_images(const char *filename, uint8_t *images, size_t count) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fseek(file, 16, SEEK_SET);  // Skip the header
    fread(images, sizeof(uint8_t), IMG_SIZE * count, file);
    fclose(file);
}

// Function to load labels from Fashion-MNIST
void load_labels(const char *filename, uint8_t *labels, size_t count) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fseek(file, 8, SEEK_SET);  // Skip the header
    fread(labels, sizeof(uint8_t), count, file);
    fclose(file);
}

// Function to filter and balance dataset by class
void filter_and_balance_dataset(uint8_t *images, uint8_t *labels, 
                                Dataset *tshirt_dataset, Dataset *ankleboot_dataset, 
                                size_t total_images) {
    size_t tshirt_index = 0, ankleboot_index = 0;

    for (size_t i = 0; i < total_images; i++) {
        if (labels[i] == 0 && tshirt_index < 500) { // T-shirt = 0
            memcpy(&tshirt_dataset->images[tshirt_index * IMG_SIZE], &images[i * IMG_SIZE], IMG_SIZE);
            tshirt_dataset->labels[tshirt_index] = 0;
            tshirt_index++;
        } else if (labels[i] == 9 && ankleboot_index < 500) { // Ankle boot = 9
            memcpy(&ankleboot_dataset->images[ankleboot_index * IMG_SIZE], &images[i * IMG_SIZE], IMG_SIZE);
            ankleboot_dataset->labels[ankleboot_index] = 1;
            ankleboot_index++;
        }
        if (tshirt_index == 500 && ankleboot_index == 500) break;
    }

    tshirt_dataset->count = tshirt_index;
    ankleboot_dataset->count = ankleboot_index;
}

// Function to split dataset into training and testing sets
void split_balanced_dataset(Dataset *tshirt_dataset, Dataset *ankleboot_dataset, 
                            Dataset *train_set, Dataset *test_set) {
    size_t train_tshirt_count = 400;
    size_t train_ankleboot_count = 400;
    size_t test_tshirt_count = tshirt_dataset->count - train_tshirt_count;
    size_t test_ankleboot_count = ankleboot_dataset->count - train_ankleboot_count;

    // Allocate memory for training and testing sets
    train_set->images = (uint8_t *)malloc((train_tshirt_count + train_ankleboot_count) * IMG_SIZE * sizeof(uint8_t));
    train_set->labels = (uint8_t *)malloc((train_tshirt_count + train_ankleboot_count) * sizeof(uint8_t));
    test_set->images = (uint8_t *)malloc((test_tshirt_count + test_ankleboot_count) * IMG_SIZE * sizeof(uint8_t));
    test_set->labels = (uint8_t *)malloc((test_tshirt_count + test_ankleboot_count) * sizeof(uint8_t));

    // Add T-shirts to training and testing sets
    memcpy(train_set->images, tshirt_dataset->images, train_tshirt_count * IMG_SIZE * sizeof(uint8_t));
    memcpy(train_set->labels, tshirt_dataset->labels, train_tshirt_count * sizeof(uint8_t));
    memcpy(test_set->images, tshirt_dataset->images + train_tshirt_count * IMG_SIZE, test_tshirt_count * IMG_SIZE * sizeof(uint8_t));
    memcpy(test_set->labels, tshirt_dataset->labels + train_tshirt_count, test_tshirt_count * sizeof(uint8_t));

    // Add Ankle boots to training and testing sets
    memcpy(train_set->images + train_tshirt_count * IMG_SIZE, ankleboot_dataset->images, train_ankleboot_count * IMG_SIZE * sizeof(uint8_t));
    memcpy(train_set->labels + train_tshirt_count, ankleboot_dataset->labels, train_ankleboot_count * sizeof(uint8_t));
    memcpy(test_set->images + test_tshirt_count * IMG_SIZE, ankleboot_dataset->images + train_ankleboot_count * IMG_SIZE, test_ankleboot_count * IMG_SIZE * sizeof(uint8_t));
    memcpy(test_set->labels + test_tshirt_count, ankleboot_dataset->labels + train_ankleboot_count, test_ankleboot_count * sizeof(uint8_t));

    train_set->count = train_tshirt_count + train_ankleboot_count;
    test_set->count = test_tshirt_count + test_ankleboot_count;
}

// Function to save a dataset to binary files
void save_dataset(const char *image_file, const char *label_file, Dataset *dataset) {
    FILE *image_fp = fopen(image_file, "wb");
    FILE *label_fp = fopen(label_file, "wb");
    if (!image_fp || !label_fp) {
        fprintf(stderr, "Error opening output files.\n");
        exit(EXIT_FAILURE);
    }

    fwrite(dataset->images, sizeof(uint8_t), dataset->count * IMG_SIZE, image_fp);
    fwrite(dataset->labels, sizeof(uint8_t), dataset->count, label_fp);

    fclose(image_fp);
    fclose(label_fp);
}

int main() {
    size_t total_images = 60000;
    uint8_t *images = (uint8_t *)malloc(total_images * IMG_SIZE * sizeof(uint8_t));
    uint8_t *labels = (uint8_t *)malloc(total_images * sizeof(uint8_t));

    // Load data
    load_images("train-images-idx3-ubyte", images, total_images);
    load_labels("train-labels-idx1-ubyte", labels, total_images);

    // Filter dataset by class
    Dataset tshirt_dataset, ankleboot_dataset;
    tshirt_dataset.images = (uint8_t *)malloc(500 * IMG_SIZE * sizeof(uint8_t));
    tshirt_dataset.labels = (uint8_t *)malloc(500 * sizeof(uint8_t));
    ankleboot_dataset.images = (uint8_t *)malloc(500 * IMG_SIZE * sizeof(uint8_t));
    ankleboot_dataset.labels = (uint8_t *)malloc(500 * sizeof(uint8_t));
    filter_and_balance_dataset(images, labels, &tshirt_dataset, &ankleboot_dataset, total_images);

    // Split dataset into training and testing sets
    Dataset train_set, test_set;
    split_balanced_dataset(&tshirt_dataset, &ankleboot_dataset, &train_set, &test_set);

    // Save to files
    save_dataset("train_images.bin", "train_labels.bin", &train_set);
    save_dataset("test_images.bin", "test_labels.bin", &test_set);

    // Free memory
    free(images);
    free(labels);
    free(tshirt_dataset.images);
    free(tshirt_dataset.labels);
    free(ankleboot_dataset.images);
    free(ankleboot_dataset.labels);
    free(train_set.images);
    free(train_set.labels);
    free(test_set.images);
    free(test_set.labels);

    printf("Dataset preparation complete!\n");
    return 0;
}
