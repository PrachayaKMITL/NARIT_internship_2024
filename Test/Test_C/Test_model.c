#include <stdio.h>
#include <stdlib.h>

void read_jpeg_file(const char *filename) {
    // Open the file in binary mode
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    // Allocate memory for the file content
    unsigned char *buffer = (unsigned char *)malloc(file_size);
    if (buffer == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return;
    }

    // Read the file into the buffer
    size_t bytes_read = fread(buffer, 1, file_size, file);
    if (bytes_read != file_size) {
        fprintf(stderr, "Error reading file\n");
        free(buffer);
        fclose(file);
        return;
    }

    // For demonstration purposes, print the first few bytes of the JPEG file
    printf("First few bytes of %s:\n", filename);
    for (size_t i = 0; i < 10 && i < bytes_read; i++) {
        printf("%02X ", buffer[i]);
    }
    printf("\n");

    // Clean up
    free(buffer);
    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image.jpg>\n", argv[0]);
        return 1;
    }

    read_jpeg_file(argv[1]);
    return 0;
}

