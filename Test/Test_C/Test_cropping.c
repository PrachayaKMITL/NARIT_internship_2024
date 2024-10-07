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

    // Calculate stride (number of bytes per row)
    int stride = img_w * img_chan;

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

int main(int argc, char const *argv[]) {
    int im_w, im_h, im_chan;
    int crop_w, crop_h;
    int crop_size = 570;

    // Load the image
    const char *image_path = "C:\\Users\\ASUS\\Documents\\NARIT_internship_2024\\NARIT_internship_2024\\Test\\Test_C\\masked_output_image.png";
    unsigned char *image = stbi_load(image_path, &im_w, &im_h, &im_chan, 0);
    if (image == NULL) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        return 1;
    }

    // Crop the image
    unsigned char *cropped_image = crop_center(image, im_w, im_h, im_chan, crop_size, &crop_w, &crop_h);
    if (cropped_image == NULL) {
        stbi_image_free(image);
        return 1;
    }

    // Save the cropped image
    const char *output_image_file = "crop_image.png";
    if (stbi_write_png(output_image_file, crop_w, crop_h, im_chan, cropped_image, crop_w * im_chan)) {
        printf("Cropped image saved as: %s\n", output_image_file);
    } else {
        printf("Failed to save cropped image\n");
    }

    // Free allocated memory
    stbi_image_free(image);
    free(cropped_image);

    return 0;
}
