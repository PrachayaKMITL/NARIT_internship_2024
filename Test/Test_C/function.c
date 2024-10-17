#include <stdio.h>
#include <stdlib.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include "function.h"

int main() {
    int img_w, img_h, img_c;  // Variables to store image width, height, and channel count
    int mask_w, mask_h, mask_c;
    int crop_size = 570;       // Desired size for cropping
    int crop_w, crop_h;        // Variables to store the width and height of the cropped image

    // Correcting string literals: use double quotes for strings
    const char *image_path = "C:\\Users\\ASUS\\Documents\\NARIT_internship_data\\All_sky_camera_Astropark_Chaingmai\2024-06\2024-06-06\638532351071115149.png";
    const char *mask_path = "C:\\Users\\ASUS\\Documents\\NARIT_internship_2024\\NARIT_internship_2024\\masks\\Domestic observatories\\mask_Astropark.png";

    // Load the image
    unsigned char *image = stbi_load(image_path, &img_w, &img_h, &img_c, 0);
    // Load the mask
    unsigned char *mask = stbi_load(mask_path, &mask_w, &mask_h, &mask_c, 0);
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

    // Save the cropped image
    const char *output_image_path = "crop_image.png";
    if (stbi_write_png(output_image_path, crop_w, crop_h, img_c, crop_image, crop_w * img_c)) {
        printf("Cropped image saved as: %s\n", output_image_path);
    } else {
        printf("Failed to save cropped image\n");
    }

    // Free allocated memory
    free(masked_image);           // Free the masked image memory
    free(crop_image);             // Free the cropped image memory
    stbi_image_free(image);       // Free the original image memory
    stbi_image_free(mask);        // Free the mask memory

    return 0;  // Indicate successful completion
}
