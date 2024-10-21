#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define BUFFER_SIZE 1024
#define NUM_FEATURES 12

void parse_parameters(const char *filename, double mean[], double scale[]) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Could not open file %s.\n", filename);
        exit(EXIT_FAILURE);
    }

    char buffer[BUFFER_SIZE];
    while (fgets(buffer, BUFFER_SIZE, file)) {
        if (strstr(buffer, "\"mean\":")) {
            for (int i = 0; i < NUM_FEATURES; i++) {
                fgets(buffer, BUFFER_SIZE, file);
                sscanf(buffer, " %lf,", &mean[i]); // Read mean values
            }
        }
        if (strstr(buffer, "\"scale\":")) {
            for (int i = 0; i < NUM_FEATURES; i++) {
                fgets(buffer, BUFFER_SIZE, file);
                sscanf(buffer, " %lf,", &scale[i]); // Read scale values
            }
        }
    }

    fclose(file);
}
void standard_Scaler(double* data, double* scaled_data,int size, double mean[NUM_FEATURES], double std[NUM_FEATURES]){
    for (int i = 0; i < size; i++){
        scaled_data[i] = (data[i] - mean[i]) / std[i];
    }
}
int main() {
    double mean[NUM_FEATURES];
    double scale[NUM_FEATURES]; // Changed std to scale
    double data[] = {28.353563, 0.541332, 0.881711, 0.661814, 0.995837, 0.437998, 36.749763, 21.832547, 36.749763, 2.091780, 67.676135, -14.917216};
    double scaled_Data[NUM_FEATURES];

    int size = sizeof(data) / sizeof(data[0]); // Corrected size calculation
    // Parse the parameters from the text file
    parse_parameters("C:\\Users\\ASUS\\Documents\\NARIT_internship_2024\\NARIT_internship_2024\\models\\Scaler\\scaler_params.txt", mean, scale);
    standard_Scaler(data, scaled_Data, size, mean, scale); // Use mean and scale

    // Optionally, print the scaled data for verification
    for (int i = 0; i < size; i++) {
        printf("Data[%d]: %lf\n", i, data[i]);
    }
    for (int i = 0; i < size; i++) {
        printf("Scaled Data[%d]: %lf\n", i, scaled_Data[i]);
    }

    return EXIT_SUCCESS;
}