#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stb_image.h>
#include <stb_image_write.h>

#define LEVELS 256  // Assuming GLCM is for 8-bit grayscale image

// Function to convert an RGB image to grayscale
unsigned char* convert_to_grayscale(unsigned char *image, int width, int height, int channels) {
    unsigned char *grayscale_image = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    if (grayscale_image == NULL) {
        printf("Failed to allocate memory for grayscale image\n");
        return NULL;
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width * channels + j * channels;
            unsigned char r = image[idx];
            unsigned char g = image[idx + 1];
            unsigned char b = image[idx + 2];
            
            // Convert to grayscale using luminance formula
            unsigned char gray = (unsigned char)(0.3 * r + 0.59 * g + 0.11 * b);
            grayscale_image[i * width + j] = gray;
        }
    }

    return grayscale_image;
}

// Function to compute the GLCM matrix
void compute_glcm(unsigned char *image, int rows, int cols, int dx, int dy, double P[LEVELS][LEVELS]) {
    int i, j;

    // Initialize the GLCM matrix to 0
    for (i = 0; i < LEVELS; i++) {
        for (j = 0; j < LEVELS; j++) {
            P[i][j] = 0.0;
        }
    }

    // Compute the GLCM matrix
    for (i = 0; i < rows - dy; i++) {
        for (j = 0; j < cols - dx; j++) {
            int pixel_val = image[i * cols + j];
            int neighbor_val = image[(i + dy) * cols + (j + dx)];
            P[pixel_val][neighbor_val]++;
        }
    }
}

// Function to normalize the GLCM matrix
void normalizeGLCM(double P[LEVELS][LEVELS], int num_levels) {
    double total_sum = 0.0;
    int i, j;

    // Calculate the sum of all elements in the GLCM
    for (i = 0; i < num_levels; i++) {
        for (j = 0; j < num_levels; j++) {
            total_sum += P[i][j];
        }
    }

    // Normalize the GLCM by dividing each element by the total sum
    if (total_sum != 0) {
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                P[i][j] /= total_sum;
            }
        }
    }
}

// Function to compute properties of the GLCM
void compute_properties(double P[LEVELS][LEVELS], int num_levels, const char* prop, double* result) {
    int i, j;
    double sum = 0.0;
    double asm_val = 0.0;
    double entropy = 0.0;

    // Normalize the GLCM matrix
    double total_sum = 0.0;
    for (i = 0; i < num_levels; i++) {
        for (j = 0; j < num_levels; j++) {
            total_sum += P[i][j];
        }
    }
    if (total_sum == 0) total_sum = 1.0; // Avoid division by zero

    // Normalize the GLCM
    for (i = 0; i < num_levels; i++) {
        for (j = 0; j < num_levels; j++) {
            P[i][j] /= total_sum;
        }
    }

    if (strcmp(prop, "contrast") == 0) {
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                sum += P[i][j] * (i - j) * (i - j);
            }
        }
        *result = sum;

    } else if (strcmp(prop, "energy") == 0) {
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                asm_val += P[i][j] * P[i][j];
            }
        }
        *result = sqrt(asm_val);

    } else if (strcmp(prop, "dissimilarity") == 0) {
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                sum += P[i][j] * fabs(i - j);
            }
        }
        *result = sum;
    }
}

// Main function
int main() {
    double features[]
    double P[LEVELS][LEVELS];
    int img_w, img_h, img_c;
    const char *image_path = "C:\\Users\\ASUS\\Documents\\NARIT_internship_2024\\NARIT_internship_2024\\Test\\Test_C\\masked_output_image.png";
    
    // Load image
    unsigned char *image = stbi_load(image_path, &img_w, &img_h, &img_c, 0);
    if (image == NULL) {
        printf("Error loading image: %s\n", stbi_failure_reason());
        return 1;
    }

    int dx = 1, dy = 0; // Offset for GLCM computation

    // Convert to grayscale
    unsigned char *gray_image = convert_to_grayscale(image, img_w, img_h, img_c);
    if (gray_image == NULL) {
        stbi_image_free(image);
        return 1;
    }

    // Compute the GLCM matrix
    compute_glcm(gray_image, img_h, img_w, dx, dy, P);
    normalizeGLCM(P, LEVELS);

    // Compute and print properties
    double contrast_result;
    compute_properties(P, LEVELS, "contrast", &contrast_result);
    printf("Contrast: %f\n", contrast_result);

    double energy_result;
    compute_properties(P, LEVELS, "energy", &energy_result);
    printf("Energy: %f\n", energy_result);

    double dissimilarity_result;
    compute_properties(P, LEVELS, "dissimilarity", &dissimilarity_result);
    printf("Dissimilarity: %f\n", dissimilarity_result);

    

    // Free image memory
    free(gray_image);
    stbi_image_free(image);

    return 0;
}
