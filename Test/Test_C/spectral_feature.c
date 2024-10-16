#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stb_image.h>
#include <stb_image_write.h>

void calculate_image_statistics(const char *image_path) {
    int img_w, img_h, img_c;  // Variables for image dimensions and channel count

    // Load the image
    unsigned char *image = stbi_load(image_path, &img_w, &img_h, &img_c, 0);
    if (image == NULL || img_c != 3) {
        printf("Error loading image: %s\n", stbi_failure_reason());
        return;  // Exit if loading fails or if not RGB
    }

    // Allocate memory for separate channels
    unsigned char *red_channel = (unsigned char *)malloc(img_w * img_h);
    unsigned char *green_channel = (unsigned char *)malloc(img_w * img_h);
    unsigned char *blue_channel = (unsigned char *)malloc(img_w * img_h);

    if (!red_channel || !green_channel || !blue_channel) {
        printf("Error allocating memory for channels.\n");
        stbi_image_free(image);
        return;  // Exit if memory allocation fails
    }

    // Separate the channels
    for (int y = 0; y < img_h; y++) {
        for (int x = 0; x < img_w; x++) {
            // Calculate pixel index
            int pixel_index = (y * img_w + x) * img_c;

            // Assign each channel
            red_channel[y * img_w + x] = image[pixel_index];       // Red
            green_channel[y * img_w + x] = image[pixel_index + 1]; // Green
            blue_channel[y * img_w + x] = image[pixel_index + 2];  // Blue
        }
    }

    // Calculate averages
    double red_sum = 0, blue_sum = 0;
    for (int i = 0; i < img_w * img_h; i++) {
        red_sum += red_channel[i];
        blue_sum += blue_channel[i];
    }
    double avg_red = red_sum / (img_w * img_h);
    double avg_blue = blue_sum / (img_w * img_h);

    // Calculate skewness of the blue channel
    // Calculate skewness of the blue channel
    double skewness = 0;
    double blue_variance_sum = 0;
    double n = img_w*img_h;
    for (int i = 0; i < n; i++) {
        skewness += pow((blue_channel[i] - avg_blue), 3);
        blue_variance_sum += pow((blue_channel[i] - avg_blue), 2);
    }
    double std_dev = sqrt(blue_variance_sum / n);
    skewness = (n / ((n - 1) * (n - 2))) * (skewness / pow(std_dev, 3));

    // Calculate average difference between red and blue channels
    double avg_diff = 0;
    for (int i = 0; i < img_w * img_h; i++) {
        avg_diff += red_channel[i] - blue_channel[i];
    }
    avg_diff /= (img_w * img_h);

    // Print results
    printf("Average Red Channel Value: %f\n", avg_red);
    printf("Average Blue Channel Value: %f\n", avg_blue);
    printf("Skewness of Blue Channel: %f\n", skewness);
    printf("Standard Deviation of Blue Channel: %f\n", std_dev);
    printf("Average Difference between Red and Blue Channels: %f\n", avg_diff);

    // Free allocated memory
    free(red_channel);
    free(green_channel);
    free(blue_channel);
    stbi_image_free(image);
}

int main() {
    const char *image_path = "C:\\Users\\ASUS\\Documents\\NARIT_internship_data\\All_sky_camera_Astropark_Chaingmai\\2024-08\\2024-08-06\\638584979539730949.png"; // Replace with your image path
    calculate_image_statistics(image_path);
    return 0;
}
