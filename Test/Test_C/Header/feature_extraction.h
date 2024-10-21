#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define LEVELS 256 // Corrected definition of LEVELS

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

// Define type definition
typedef struct {
    double skewness;
    double std_dev;
    double avg_red;
    double avg_blue;
    double avg_diff;
} ImageStatistics;

ImageStatistics calculate_image_statistics(unsigned char *image, int img_w, int img_h, int img_c) {
    // Allocate memory for separate channels
    unsigned char *red_channel = (unsigned char *)malloc(img_w * img_h);
    unsigned char *green_channel = (unsigned char *)malloc(img_w * img_h); // Corrected this line
    unsigned char *blue_channel = (unsigned char *)malloc(img_w * img_h);

    if (!red_channel || !green_channel || !blue_channel) {
        printf("Error allocating memory for channels.\n");
        // Free any allocated memory before returning
        free(red_channel);
        free(green_channel);
        free(blue_channel);
        // Return a default value or handle the error appropriately
        ImageStatistics stats = {0, 0, 0, 0, 0};
        return stats;
    }

    // Separate the channels
    for (int y = 0; y < img_h; y++) {
        for (int x = 0; x < img_w; x++) {
            // Calculate pixel index
            int pixel_index = (y * img_w + x) * img_c;
            red_channel[y * img_w + x] = image[pixel_index];
            green_channel[y * img_w + x] = image[pixel_index + 1];
            blue_channel[y * img_w + x] = image[pixel_index + 2];
        }
    }

    // Calculate average for each channel
    double sum_red = 0.0, sum_green = 0.0, sum_blue = 0.0;

    for (int i = 0; i < img_w * img_h; i++) {
        sum_red += red_channel[i];
        sum_green += green_channel[i];
        sum_blue += blue_channel[i];
    }

    double avg_red = sum_red / (img_w * img_h);
    double avg_blue = sum_blue / (img_w * img_h);

    // Calculate std deviation
    double var_red = 0.0, var_green = 0.0, var_blue = 0.0;

    for (int i = 0; i < img_w * img_h; i++) {
        var_red += (red_channel[i] - avg_red) * (red_channel[i] - avg_red);
        var_blue += (blue_channel[i] - avg_blue) * (blue_channel[i] - avg_blue);
    }
    double std_dev_blue = sqrt(var_blue / (img_w * img_h));

    // Calculate skewness
    double skewness_red = 0.0, skewness_green = 0.0, skewness_blue = 0.0;

    for (int i = 0; i < img_w * img_h; i++) {
        skewness_blue += pow((blue_channel[i] - avg_blue) / std_dev_blue, 3);
    }

    skewness_red /= img_w * img_h;
    skewness_green /= img_w * img_h;
    skewness_blue = ((img_w * img_h) / ((double)((img_w * img_h) - 1) * ((img_w * img_h)- 2))) * skewness_blue;

    // Calculate average difference
    double avg_diff = avg_red - avg_blue;

    // Free allocated memory
    free(red_channel);
    free(green_channel);
    free(blue_channel);

    // Return statistics
    ImageStatistics stats = {
        .skewness = skewness_blue, // You can choose which skewness to return
        .std_dev = std_dev_blue, // Same for std_dev
        .avg_red = avg_red,
        .avg_blue = avg_blue,
        .avg_diff = avg_diff
    };

    return stats;
}