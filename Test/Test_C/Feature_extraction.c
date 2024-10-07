#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stb_image.h>
//#include "C:\Users\ASUS\Documents\NARIT_internship_2024\C_source_code\dep\Microsoft.ML.OnnxRuntime.1.16.0\build\native\include\onnxruntime_c_api.h"

#define LEVELS 256  // Assuming GLCM is for 8-bit grayscale image

#define IMAGE_WIDTH 100
#define IMAGE_HEIGHT 100

int image_width,image_height,channels;
// Function to generate a test image
void generate_test_image(int image[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    // Fill the image with a simple pattern
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            if ((i / 10) % 2 == 0) {
                // Create horizontal stripes
                image[i][j] = (j / 10) % 2 == 0 ? 255 : 0;  // Alternating black and white
            } else {
                // Create vertical stripes
                image[i][j] = (i / 10) % 2 == 0 ? 255 : 0;  // Alternating black and white
            }
        }
    }
}

// Function to compute GLCM from the image
void compute_glcm(int image[100][100], int rows, int cols, int dx, int dy, double P[LEVELS][LEVELS]) {
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
            int pixel_val = image[i][j];
            int neighbor_val = image[i + dy][j + dx];
            P[pixel_val][neighbor_val]++;
        }
    }
}

// Function to compute properties of the GLCM
void compute_properties(double P[LEVELS][LEVELS], int num_levels, char* prop, double* result) {
    int i, j;
    double sum = 0.0;
    double mean = 0.0;
    double variance = 0.0;
    double asm_val = 0.0;
    double energy = 0.0;
    double entropy = 0.0;
    double correlation = 0.0;
    double std_i = 0.0, std_j = 0.0;

    // Calculate the total sum to normalize the GLCM
    double total_sum = 0.0;
    for (i = 0; i < num_levels; i++) {
        for (j = 0; j < num_levels; j++) {
            total_sum += P[i][j];
        }
    }
    if (total_sum == 0) total_sum = 1.0; // Avoid division by zero

    // Normalize the GLCM matrix
    for (i = 0; i < num_levels; i++) {
        for (j = 0; j < num_levels; j++) {
            P[i][j] /= total_sum;
        }
    }

    if (strcmp(prop, "contrast") == 0) {
        // Contrast: sum of (i-j)^2 * P[i,j]
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                sum += P[i][j] * (i - j) * (i - j);
            }
        }
        *result = sum;

    } else if (strcmp(prop, "dissimilarity") == 0) {
        // Dissimilarity: sum of |i-j| * P[i,j]
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                sum += P[i][j] * fabs(i - j);
            }
        }
        *result = sum;

    } else if (strcmp(prop, "homogeneity") == 0) {
        // Homogeneity: sum of P[i,j] / (1 + (i-j)^2)
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                sum += P[i][j] / (1.0 + (i - j) * (i - j));
            }
        }
        *result = sum;

    } else if (strcmp(prop, "ASM") == 0) {
        // ASM: sum of P[i,j]^2
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                asm_val += P[i][j] * P[i][j];
            }
        }
        *result = asm_val;

    } else if (strcmp(prop, "energy") == 0) {
        // Energy: sqrt(ASM)
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                asm_val += P[i][j] * P[i][j];
            }
        }
        energy = sqrt(asm_val);
        *result = energy;

    } else if (strcmp(prop, "entropy") == 0) {
        // Entropy: -sum of P[i,j] * log(P[i,j])
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                if (P[i][j] > 0) {
                    entropy += P[i][j] * log(P[i][j]);
                }
            }
        }
        *result = -entropy;

    } else if (strcmp(prop, "mean") == 0) {
        // Mean: sum of i * P[i,j]
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                mean += i * P[i][j];
            }
        }
        *result = mean;

    } else if (strcmp(prop, "variance") == 0) {
        // Variance: sum of P[i,j] * (i - mean)^2
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                mean += i * P[i][j];
            }
        }
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                variance += P[i][j] * (i - mean) * (i - mean);
            }
        }
        *result = variance;

    } else if (strcmp(prop, "correlation") == 0) {
        // Correlation: sum of (i - mean_i)(j - mean_j)P[i,j] / (std_i * std_j)
        double mean_i = 0.0, mean_j = 0.0;
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                mean_i += i * P[i][j];
                mean_j += j * P[i][j];
            }
        }
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                std_i += P[i][j] * pow(i - mean_i, 2);
                std_j += P[i][j] * pow(j - mean_j, 2);
            }
        }
        std_i = sqrt(std_i);
        std_j = sqrt(std_j);

        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                correlation += (i - mean_i) * (j - mean_j) * P[i][j];
            }
        }
        *result = correlation / (std_i * std_j);
    }
}

// Example usage
int main() {
    int image[IMAGE_HEIGHT][IMAGE_WIDTH];

    generate_test_image(image);
    double P[LEVELS][LEVELS];
    int rows = 100, cols = 100;
    int dx = 1, dy = 0; // Offset for GLCM computation

    // Compute the GLCM matrix
    compute_glcm(image, rows, cols, dx, dy, P);

    // Compute a property (e.g., contrast)
    double contrast_result;
    compute_properties(P, LEVELS, "contrast", &contrast_result);
    printf("Contrast: %f\n", contrast_result);

    // Compute another property (e.g., energy)
    double energy_result;
    compute_properties(P, LEVELS, "energy", &energy_result);
    printf("Energy: %f\n", energy_result);

    double mean_result;
    compute_properties(P,LEVELS,"mean",&mean_result);
    printf("Mean %f\n", mean_result);

    return 0;
}
