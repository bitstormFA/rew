#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_NO_FAILURE_STRINGS
#include "stb_image.h"

static unsigned char *stbi_load_from_memory_wrapper(
    unsigned char const *buffer, int len,
    int *x, int *y, int *channels_in_file, int desired_channels) {
    return stbi_load_from_memory(buffer, len, x, y, channels_in_file, desired_channels);
}

static void stbi_free_wrapper(void *data) {
    stbi_image_free(data);
}
