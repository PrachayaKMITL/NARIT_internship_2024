#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <stb_image.h>
#include <stb_image_write.h>

// Function to crop the center of the image with stride consideration
unsigned char* crop_center(unsigned char *img, int img_w, int img_h, int img_chan, int crop_size, int *out_w, int *out_h) {
    // Calculate the starting coordinates for cropping
    int start_x = (img_w - crop_size) / 2;
    int start_y = (img_h - crop_size) / 2;

    // Calculate stride (number of bytes per row) which might include padding
    int stride = img_w * img_chan;

    // Debugging: Log information about image cropping
    printf("Cropping image at center: (%d, %d), crop size: %d, original size: %dx%d\n", start_x, start_y, crop_size, img_w, img_h);

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
                    img[((start_y + j) * stride + (start_x + i) * img_chan) + c];
            }
        }
    }

    // Set output dimensions
    *out_w = crop_size;
    *out_h = crop_size;

    return cropped_img;
}

// Masking function with dimension checks
unsigned char* masking(unsigned char *image, unsigned char *mask, int img_w, int img_h, int img_chan) {
    // Debugging: Log information about mask and image dimensions
    printf("Applying mask: Image size: %dx%d, Mask size: %dx%d\n", img_w, img_h, img_w, img_h);

    // Apply the mask
    for (int y = 0; y < img_h; y++) {
        for (int x = 0; x < img_w; x++) {
            // Get the mask value (assuming single-channel mask)
            unsigned char mask_value = mask[y * img_w + x];

            // Apply mask to R, G, B channels of the RGB image
            for (int c = 0; c < img_chan; c++) {
                image[(y * img_w + x) * img_chan + c] =
                    (image[(y * img_w + x) * img_chan + c] * mask_value) / 255;
            }
        }
    }

    return image; 
}

int main() {
    // Initialize image dimensions
    int img_w, img_h, img_chan;

    // Load image and mask from paths
    const char *image_path = "C:\\Users\\ASUS\\Documents\\NARIT_internship_data\\All_sky_camera_Astropark_Chaingmai\\2024-06\\2024-06-18\\638542744942162902.png";
    const char *mask_path = "C:\\Users\\ASUS\\Documents\\NARIT_internship_2024\\NARIT_internship_2024\\masks\\Domestic observatories\\mask_Astropark.png";
    unsigned char *mask = stbi_load(mask_path, &img_w, &img_h, &img_chan, 1); // Load mask as single channel
    unsigned char *img = stbi_load(image_path, &img_w, &img_h, &img_chan, 0); // Load image with original channels
    
    if (img == NULL || mask == NULL) {
        fprintf(stderr, "Error loading image or mask: %s\n", stbi_failure_reason());
        return 1; // Exit if loading fails
    }

    // Crop the center of the image
    int cropped_w, cropped_h;
    unsigned char *cropped_img = crop_center(img, img_w, img_h, img_chan, 570, &cropped_w, &cropped_h);
    
    if (cropped_img == NULL) {
        fprintf(stderr, "Error during cropping.\n");
        stbi_image_free(img);
        stbi_image_free(mask);
        return 1;
    }

    // Apply mask to the cropped image
    unsigned char *final_img = masking(cropped_img, mask, cropped_w, cropped_h, img_chan);
    
    if (final_img != NULL) {
        // Save the cropped image to a file
        const char *output_path = "cropped_image_with_mask.png";
        if (stbi_write_png(output_path, cropped_w, cropped_h, img_chan, final_img, cropped_w * img_chan)) {
            printf("Cropped image with mask saved to %s\n", output_path);
        } else {
            fprintf(stderr, "Error saving cropped image.\n");
        }

        // Free the final image memory
        free(final_img);
    }

    // Free the original image memory
    stbi_image_free(img);
    stbi_image_free(mask);
    stbi_image_free(cropped_img);

    return 0;
}
