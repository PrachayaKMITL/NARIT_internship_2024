#include <stdio.h>
#include <stdlib.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include "Header/Preprocessing.h"
#include "Header/feature_extraction.h"

#define LEVELS 256

int main() {
    double P[LEVELS][LEVELS];
    int img_w, img_h, img_c;  // Variables to store image width, height, and channel count
    int mask_w, mask_h, mask_c;
    //int crop_size = 570;       // Desired size for cropping
    //int crop_w, crop_h;        // Variables to store the width and height of the cropped image

    // Correcting string literals: use double quotes for strings
    const char *image_path = "C:\\Users\\ASUS\\Documents\\NARIT_internship_data\\Dataset\\image_data_Astropark\\image_data_Day\\Clear\\638545284152007058.png";
    const char *mask_path = "C:\\Users\\ASUS\\Documents\\NARIT_internship_2024\\NARIT_internship_2024\\masks\\Domestic observatories\\mask_Astropark.png";
    //char image_path[200];
    //char mask_path[100];

    //printf("Enter image path : ");
    //getchar();
    //scanf("%s", image_path);
    

    //printf("Enter mask path : ");
    //getchar();
    //scanf("%s", &mask_path);

    // Load the image
    unsigned char *image = stbi_load(image_path, &img_w, &img_h, &img_c, 0);
    // Load the mask
    unsigned char *mask = stbi_load(mask_path, &mask_w, &mask_h, &mask_c, 0);
    unsigned char *masked_image = masking(image, mask, img_w, img_h, img_c);
    ImageStatistics stat = calculate_image_statistics(masked_image,img_w,img_h,img_c);
    printf("Intensity %lf\n",stat.avg_blue);
    printf("Skewness %lf\n",stat.skewness);
    printf("Std_dev %lf\n",stat.std_dev);
    printf("Different (R-B) %lf\n",stat.avg_diff);
    printf("Red channel %lf\n",stat.avg_red);
    printf("Blue channel %lf\n\n",stat.avg_blue);

    // Crop the masked image
    unsigned char *gray_image = convert_to_grayscale(masked_image, img_w, img_h, img_c);
    int dx = 1;
    int ang = 0;

    compute_glcm(gray_image,img_w,img_h,dx,ang,P);
    normalizeGLCM(P,LEVELS);

    double contrast_result;
    compute_properties(P, LEVELS, "contrast", &contrast_result);
    printf("Contrast: %f\n", contrast_result);

    double dissimilarity_result;
    compute_properties(P, LEVELS, "dissimilarity", &dissimilarity_result);
    printf("Dissimilarity: %f\n", dissimilarity_result);

    double homogeneity_result;
    compute_properties(P, LEVELS, "homogeneity", &homogeneity_result);
    printf("Homogeneity: %f\n", homogeneity_result);

    double correlation_result;
    compute_properties(P, LEVELS, "correlation", &correlation_result);
    printf("Correlation: %f\n", correlation_result);

    double energy_result;
    compute_properties(P, LEVELS, "energy", &energy_result);
    printf("Energy: %f\n", energy_result);

    double asm_result;
    compute_properties(P, LEVELS, "ASM", &asm_result);
    printf("ASM: %f\n", asm_result); 

    if (gray_image == NULL) {
        printf("Error cropping image.\n");
        stbi_image_free(image);      // Free the original image memory
        stbi_image_free(mask);       // Free the mask memory
        free(masked_image);          // Free the masked image memory
        return 1;  // Exit if cropping fails
    }
    // Free allocated memory
    free(masked_image);           // Free the masked image memory
    free(gray_image);             // Free the cropped image memory
    stbi_image_free(image);       // Free the original image memory
    stbi_image_free(mask);        // Free the mask memory

    return 0;  // Indicate successful completion
}
