#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>

int main() {
    int width_rgb, height_rgb, channels_rgb;
    int width_mask, height_mask, channels_mask;

    // Load RGB image
    const char *rgb_image_file = "C:\\Users\\ASUS\\Documents\\NARIT_internship_data\\All_sky_camera_Astropark_Chaingmai\\2024-08\\2024-08-06\\638584979539730949.png";
    unsigned char *rgb_image = stbi_load(rgb_image_file, &width_rgb, &height_rgb, &channels_rgb, 0);
    if (!rgb_image) {
        printf("Failed to load RGB image: %s\n", rgb_image_file);
        return 1;
    }

    // Load binary mask image (expected to be single-channel, grayscale)
    const char *mask_image_file = "C:\\Users\\ASUS\\Documents\\NARIT_internship_2024\\NARIT_internship_2024\\masks\\Domestic observatories\\mask_Astropark.png";
    unsigned char *mask_image = stbi_load(mask_image_file, &width_mask, &height_mask, &channels_mask, 0);
    if (!mask_image) {
        printf("Failed to load mask image: %s\n", mask_image_file);
        stbi_image_free(rgb_image);
        return 1;
    }

    // Ensure both images have the same dimensions
    if (width_rgb != width_mask || height_rgb != height_mask) {
        printf("Error: Image dimensions do not match\n");
        stbi_image_free(rgb_image);
        stbi_image_free(mask_image);
        return 1;
    }

    // Apply the binary mask to each RGB channel
    for (int y = 0; y < height_rgb; y++) {
        for (int x = 0; x < width_rgb; x++) {
            // Get the mask value (single channel, so only 1 value per pixel)
            unsigned char mask_value = mask_image[y * width_mask + x];

            // Apply mask to R, G, B channels of the RGB image
            for (int c = 0; c < channels_rgb; c++) {
                rgb_image[(y * width_rgb + x) * channels_rgb + c] =
                    (rgb_image[(y * width_rgb + x) * channels_rgb + c] * mask_value) / 255;
            }
        }
    }

    // Save the masked image as a PNG
    const char *output_image_file = "masked_output_image.png";
    if (stbi_write_png(output_image_file, width_rgb, height_rgb, channels_rgb, rgb_image, width_rgb * channels_rgb)) {
        printf("Masked image saved as: %s\n", output_image_file);
    } else {
        printf("Failed to save masked image\n");
    }

    // Free the images
    stbi_image_free(rgb_image);
    stbi_image_free(mask_image);

    return 0;
}


