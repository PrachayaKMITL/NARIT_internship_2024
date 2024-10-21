#include <stdio.h>
#include <stdlib.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include "function.h"
#include "feature_extraction.h"

int main() {
    int img_w, img_h, img_c;  // Variables to store image width, height, and channel count
    int mask_w, mask_h, mask_c;
    int crop_size = 570;       // Desired size for cropping
    int crop_w, crop_h;        // Variables to store the width and height of the cropped image

    // Correcting string literals: use double quotes for strings
    const char *image_path = "C:\\Users\\ASUS\\Documents\\NARIT_internship_data\\All_sky_camera_Astropark_Chaingmai\\2024-06\\2024-06-07\\638533227680865742.png";
    const char *mask_path = "C:\\Users\\ASUS\\Documents\\NARIT_internship_2024\\NARIT_internship_2024\\masks\\Domestic observatories\\mask_Astropark.png";

    // Load the image
    unsigned char *image = stbi_load(image_path, &img_w, &img_h, &img_c, 0);
    if (image == NULL) {
        printf("Error loading image.\n");
        return 1;
    }

    // Load the mask
    unsigned char *mask = stbi_load(mask_path, &mask_w, &mask_h, &mask_c, 0);
    if (mask == NULL) {
        printf("Error loading mask.\n");
        stbi_image_free(image);
        return 1;
    }

    // Apply the mask to the image
    unsigned char *masked_image = masking(image, mask, img_w, img_h, img_c);

    // Crop the masked image
    unsigned char *crop_image = crop_center(masked_image, img_w, img_h, img_c, crop_size, &crop_w, &crop_h);
    if (crop_image == NULL) {
        printf("Error cropping image.\n");
        stbi_image_free(image);      // Free the original image memory
        stbi_image_free(mask);       // Free the mask memory
        free(masked_image);          // Free the masked image memory
        return 1;  // Exit if cropping fails
    }

    // Compute GLCM matrix
    double GLCM[256][256];  // Using double as the GLCM values are normalized later
    int dx = 1, dy = 0;     // GLCM parameters (distance = 1, angle = 0)
    compute_glcm(crop_image, crop_h, crop_w, dx, dy, GLCM);

    // Compute and print contrast property
    double contrast_result = 0.0;
    compute_properties(GLCM, 256, "contrast", &contrast_result);
    printf("Contrast: %.3f\n", contrast_result);

    // Free allocated memory
    free(masked_image);           // Free the masked image memory
    free(crop_image);             // Free the cropped image memory
    stbi_image_free(image);       // Free the original image memory
    stbi_image_free(mask);        // Free the mask memory

    return 0;  // Indicate successful completion
}