#ifndef THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_SUPER_RESOLUTION_IMAGE_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_SUPER_RESOLUTION_IMAGE_UTILS_H_

#include <string>

namespace ImageUtils {

unsigned char* LoadImage(const std::string& file_path, int& width, int& height, int& channels_in_file, int desired_channels);
void FreeImageData(unsigned char* data);
bool SaveImage(const std::string& file_path, int width, int height, int channels, const unsigned char* data);
/**
 * @brief Resizes and normalizes image data using bilinear interpolation.
 *
 * @param in_data Pointer to the source image data (unsigned char).
 * @param in_width Width of the source image.
 * @param in_height Height of the source image.
 * @param in_channels Number of channels in the source image (e.g., 3 for RGB).
 * @param out_data Pointer to the destination buffer (float).
 * @param out_width Target width for the resized image.
 * @param out_height Target height for the resized image.
 * @param out_channels Target number of channels for the output (must be <= in_channels).
 */
void ResizeImageBilinear(const unsigned char* in_data,
                         int in_width, int in_height, int in_channels,
                         float* out_data,
                         int out_width, int out_height, int out_channels);


// Resizes an image
unsigned char* ResizeImage(const unsigned char* input_data, int input_width,
                           int input_height, int input_channels,
                           int output_width, int output_height);

}  // namespace ImageUtils

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_SUPER_RESOLUTION_IMAGE_UTILS_H_