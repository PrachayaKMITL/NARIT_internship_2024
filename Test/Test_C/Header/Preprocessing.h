#ifndef FUNCTION_H
#define FUNCTION_H

// Standard input-output library
#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// Including stb_image and stb_image_write for image processing
#include <stb_image.h>
#include <stb_image_write.h>

// Function prototypes
unsigned char* masking(unsigned char *image, unsigned char *mask, int img_w, int img_h, int img_chan);
unsigned char* crop_center(unsigned char *img, int img_w, int img_h, int img_chan, int crop_size, int *out_w, int *out_h);
unsigned char* convert_to_grayscale(unsigned char *img, int img_w, int img_h, int img_chan);

//Function declaration
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

// Function to apply a mask to an image
unsigned char* masking(unsigned char *image, unsigned char *mask, int img_w, int img_h, int img_chan) {
    // Ensure the mask dimensions match the image dimensions
    if (image == NULL || mask == NULL) {
        fprintf(stderr, "Error: Image or mask is NULL.\n");
        return NULL; // Exit the function if images are NULL
    }

    // Allocate memory for the masked image
    unsigned char *masked_image = (unsigned char *)malloc(img_w * img_h * img_chan);
    if (masked_image == NULL) {
        fprintf(stderr, "Error allocating memory for masked image.\n");
        return NULL; // Exit if memory allocation fails
    }

    // Apply the mask
    for (int y = 0; y < img_h; y++) {
        for (int x = 0; x < img_w; x++) {
            // Get the mask value (assuming single-channel mask)
            unsigned char mask_value = mask[y * img_w + x];

            // Apply mask to R, G, B channels of the RGB image
            for (int c = 0; c < img_chan; c++) {
                masked_image[(y * img_w + x) * img_chan + c] =
                    (image[(y * img_w + x) * img_chan + c] * mask_value) / 255;
            }
        }
    }
    return masked_image; // Return the masked image
}



// Function to crop the center of an image
unsigned char* crop_center(unsigned char *img, int img_w, int img_h, int img_chan, int crop_size, int *out_w, int *out_h) {
    // Calculate the starting coordinates for cropping
    int start_x = (img_w - crop_size) / 2;
    int start_y = (img_h - crop_size) / 2;

    // Allocate memory for the cropped image
    unsigned char *cropped_img = (unsigned char *)malloc(crop_size * crop_size * img_chan);
    if (cropped_img == NULL) {
        fprintf(stderr, "Error allocating memory for cropped image.\n");
        return NULL;
    }

    // Copy the pixels from the original image to the cropped image
    for (int j = 0; j < crop_size; j++) {
        for (int i = 0; i < crop_size; i++) {
            for (int c = 0; c < img_chan; c++) {
                cropped_img[(j * crop_size + i) * img_chan + c] = 
                    img[((start_y + j) * img_w + (start_x + i)) * img_chan + c]; // Fixed indexing here
            }
        }
    }

    // Set output dimensions
    *out_w = crop_size;
    *out_h = crop_size;

    return cropped_img;
}

#endif // FUNCTION_H
