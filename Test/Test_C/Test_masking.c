#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <stb_image.h>
#include <stb_image_write.h>

const char *image_path;
const char *mask_path;

// Function to apply masking to the image
void masking(const char *image_path, const char *mask_path) {
    int im_w, im_h, im_chan;
    int mask_w, mask_h, mask_chan;

    // Load the image
    unsigned char *image = stbi_load(image_path, &im_w, &im_h, &im_chan, 0);
    // Load the mask
    unsigned char *mask = stbi_load(mask_path, &mask_w, &mask_h, &mask_chan, 0);

    // Check if the images were loaded successfully
    // Ensure the mask dimensions match the image dimensions
    if (im_w != mask_w || im_h != mask_h) {
        fprintf(stderr, "Error: Image and mask dimensions do not match.\n");
        stbi_image_free(image);
        stbi_image_free(mask);
        return; // Exit the function if dimensions do not match
    }

    // Apply the mask
    for (int y = 0; y < im_h; y++) {
        for (int x = 0; x < im_w; x++) {
            // Get the mask value (assuming single-channel mask)
            unsigned char mask_value = mask[y * mask_w + x];

            // Apply mask to R, G, B channels of the RGB image
            for (int c = 0; c < im_chan; c++) {
                image[(y * im_w + x) * im_chan + c] =
                    (image[(y * im_w + x) * im_chan + c] * mask_value) / 255;
            }
        }
    }

    // Save the masked image
    const char *output_image_file = "masked_output_image.png";
    if (stbi_write_png(output_image_file, im_w, im_h, im_chan, image, im_w * im_chan)) {
        printf("Masked image saved as: %s\n", output_image_file);
    } else {
        printf("Failed to save masked image\n");
    }

    // Free allocated memory
    stbi_image_free(image);
    stbi_image_free(mask);
}

int main(int argc, char const *argv[]) {
    // Set the paths for the image and mask
    const char *image_path = "C:\\Users\\ASUS\\Documents\\NARIT_internship_data\\Dataset\\image_data_Astropark\\image_data_Day\\Clear\\638545286553562657.png";
    const char *mask_path = "C:\\Users\\ASUS\\Documents\\NARIT_internship_2024\\NARIT_internship_2024\\masks\\Domestic observatories\\mask_Astropark.png";

    // Call the masking function
    masking(image_path, mask_path);

    return 0;
}
