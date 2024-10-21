#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stb_image.h>
#include <stb_image_write.h>

#define LEVELS 256  // Assuming GLCM is for 8-bit grayscale image

// Function to compute GLCM
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

// Function to compute properties of the GLCM
void compute_properties(double P[LEVELS][LEVELS], int num_levels, const char* prop, double* result) {
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
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                sum += P[i][j] * (i - j) * (i - j);
            }
        }
        *result = sum;

    } else if (strcmp(prop, "dissimilarity") == 0) {
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                sum += P[i][j] * fabs(i - j);
            }
        }
        *result = sum;

    } else if (strcmp(prop, "homogeneity") == 0) {
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                sum += P[i][j] / (1.0 + (i - j) * (i - j));
            }
        }
        *result = sum;

    } else if (strcmp(prop, "ASM") == 0) {
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                asm_val += P[i][j] * P[i][j];
            }
        }
        *result = asm_val;

    } else if (strcmp(prop, "energy") == 0) {
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                asm_val += P[i][j] * P[i][j];
            }
        }
        energy = sqrt(asm_val);
        *result = energy;

    } else if (strcmp(prop, "entropy") == 0) {
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                if (P[i][j] > 0) {
                    entropy += P[i][j] * log(P[i][j]);
                }
            }
        }
        *result = -entropy;

    } else if (strcmp(prop, "mean") == 0) {
        for (i = 0; i < num_levels; i++) {
            for (j = 0; j < num_levels; j++) {
                mean += i * P[i][j];
            }
        }
        *result = mean;

    } else if (strcmp(prop, "variance") == 0) {
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

// Function to calculate image statistics
void calculate_image_statistics(unsigned char *image, int img_w, int img_h, int img_c) {
    // Allocate memory for separate channels
    unsigned char *red_channel = (unsigned char *)malloc(img_w * img_h);
    unsigned char *green_channel = (unsigned char *)malloc(img_w * img_h);
    unsigned char *blue_channel = (unsigned char *)malloc(img_w * img_h);

    if (!red_channel || !green_channel || !blue_channel) {
        printf("Error allocating memory for channels.\n");
        return;  // Exit if memory allocation fails
    }

    // Separate the channels
    for (int y = 0; y < img_h; y++) {
        for (int x = 0; x < img_w; x++) {
            // Calculate pixel index
            int pixel_index = (y * img_w + x) * img_c;

            // Assign each channel
            red_channel[y * img_w + x] = image[pixel_index];       // Red
            green_channel[y * img_w + x] = image[pixel_index + 1]; // Green
            blue_channel[y * img_w + x] = image[pixel_index + 2];  // Blue
        }
    }

    // Calculate averages
    double red_sum = 0, blue_sum = 0;
    for (int i = 0; i < img_w * img_h; i++) {
        red_sum += red_channel[i];
        blue_sum += blue_channel[i];
    }
    double avg_red = red_sum / (img_w * img_h);
    double avg_blue = blue_sum / (img_w * img_h);

    // Calculate skewness of the blue channel
    double skewness = 0;
    double blue_variance_sum = 0;
    double n = img_w * img_h;
    for (int i = 0; i < n; i++) {
        skewness += pow((blue_channel[i] - avg_blue), 3);
        blue_variance_sum += pow((blue_channel[i] - avg_blue), 2);
    }
    double std_dev = sqrt(blue_variance_sum / n);
    skewness = (n / ((n - 1) * (n - 2))) * (skewness / pow(std_dev, 3));

    // Calculate average difference between red and blue channels
    double avg_diff = 0;
    for (int i = 0; i < img_w * img_h; i++) {
        avg_diff += red_channel[i] - blue_channel[i];
    }
    avg_diff /= (img_w * img_h);

    // Output the statistics
    printf("Average Red Channel Value: %.3f\n", avg_red);
    printf("Average Blue Channel Value: %.3f\n", avg_blue);
    printf("Skewness of Blue Channel: %.3f\n", skewness);
    printf("Average Difference between Red and Blue Channels: %.3f\n", avg_diff);

    // Free the allocated memory
    free(red_channel);
    free(green_channel);
    free(blue_channel);
}

int main() {
    int img_w, img_h, img_c;
    const char *image_path = "C:\\Users\\ASUS\\Documents\\NARIT_internship_2024\\NARIT_internship_2024\\Test\\Test_C\\masked_output_image.png";
    unsigned char *image = stbi_load(image_path, &img_w, &img_h, &img_c, 0);
    if (!image) {
        printf("Error loading image: %s\n", stbi_failure_reason());
        return 1;
    }

    // Convert to grayscale for GLCM
    unsigned char *gray_image = (unsigned char *)malloc(img_w * img_h);
    for (int y = 0; y < img_h; y++) {
        for (int x = 0; x < img_w; x++) {
            int pixel_index = (y * img_w + x) * img_c;
            gray_image[y * img_w + x] = (unsigned char)(0.2989 * image[pixel_index] + 0.5870 * image[pixel_index + 1] + 0.1140 * image[pixel_index + 2]);
        }
    }

    // Calculate GLCM
    double GLCM[LEVELS][LEVELS];
    compute_glcm(gray_image, img_h, img_w, 1, 0, GLCM); // Horizontal GLCM
    double contrast, dissimilarity, homogeneity, asm_val, energy, entropy, mean, variance, correlation;
    
    compute_properties(GLCM, LEVELS, "contrast", &contrast);
    compute_properties(GLCM, LEVELS, "dissimilarity", &dissimilarity);
    compute_properties(GLCM, LEVELS, "homogeneity", &homogeneity);
    compute_properties(GLCM, LEVELS, "ASM", &asm_val);
    compute_properties(GLCM, LEVELS, "energy", &energy);
    compute_properties(GLCM, LEVELS, "entropy", &entropy);
    compute_properties(GLCM, LEVELS, "mean", &mean);
    compute_properties(GLCM, LEVELS, "variance", &variance);
    compute_properties(GLCM, LEVELS, "correlation", &correlation);

    // Print GLCM features
    printf("GLCM Features:\n");
    printf("Contrast: %.4f\n", contrast);
    printf("Dissimilarity: %.4f\n", dissimilarity);
    printf("Homogeneity: %.4f\n", homogeneity);
    printf("ASM: %.4f\n", asm_val);
    printf("Energy: %.4f\n", energy);
    printf("Entropy: %.4f\n", entropy);
    printf("Mean: %.4f\n", mean);
    printf("Variance: %.4f\n", variance);
    printf("Correlation: %.4f\n", correlation);

    // Calculate image statistics
    calculate_image_statistics(image, img_w, img_h, img_c);

    // Free memory
    free(image);
    free(gray_image);
    return 0;
}
