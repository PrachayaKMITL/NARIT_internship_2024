#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <string.h>

int main() {
    char image_path[100];
    printf("Enter image : ");
    scanf("%s", image_path);
    printf("\nNumber %s",image_path);
}

