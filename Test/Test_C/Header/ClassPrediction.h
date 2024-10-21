#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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