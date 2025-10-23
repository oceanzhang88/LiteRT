#include <algorithm>  // For std::min/std::max
#include <cmath>      // For std::floor
#include <new> // For std::nothrow
#include <iostream>

#include "litert/samples/super_resolution/image_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

namespace ImageUtils {

unsigned char* LoadImage(const std::string& file_path, int& width, int& height, int& channels_in_file, int desired_channels) {
    return stbi_load(file_path.c_str(), &width, &height, &channels_in_file, desired_channels);
}

void FreeImageData(unsigned char* data) {
    if (data) {
        stbi_image_free(data);
    }
}

bool SaveImage(const std::string& file_path, int width, int height, int channels, const unsigned char* data) {
    if (file_path.substr(file_path.find_last_of(".") + 1) == "png") {
        return stbi_write_png(file_path.c_str(), width, height, channels, data, width * channels) != 0;
    } else {
        // Add support for other file types if needed
        return false;
    }
}

void ResizeImageBilinear(const unsigned char* in_data,
                         int in_width, int in_height, int in_channels,
                         float* out_data,
                         int out_width, int out_height, int out_channels) {
    const float x_ratio = static_cast<float>(in_width) / static_cast<float>(out_width);
    const float y_ratio = static_cast<float>(in_height) / static_cast<float>(out_height);

    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            // Find corresponding coordinates in the original image
            float gx = (x + 0.5f) * x_ratio - 0.5f;
            float gy = (y + 0.5f) * y_ratio - 0.5f;

            // Get integer coordinates for the top-left pixel
            int gxi = static_cast<int>(std::floor(gx));
            int gyi = static_cast<int>(std::floor(gy));

            // Calculate fractional parts (weights)
            const float wx = gx - gxi;
            const float wy = gy - gyi;

            for (int c = 0; c < out_channels; ++c) {
                // Clamp coordinates to be within image bounds
                int x0 = std::max(0, gxi);
                int x1 = std::min(in_width - 1, gxi + 1);
                int y0 = std::max(0, gyi);
                int y1 = std::min(in_height - 1, gyi + 1);

                // Get pixel values from 4 neighbors
                float p00 = in_data[(y0 * in_width + x0) * in_channels + c];
                float p10 = in_data[(y0 * in_width + x1) * in_channels + c];
                float p01 = in_data[(y1 * in_width + x0) * in_channels + c];
                float p11 = in_data[(y1 * in_width + x1) * in_channels + c];

                // Interpolate along x-axis
                float interp_x0 = (1.0f - wx) * p00 + wx * p10;
                float interp_x1 = (1.0f - wx) * p01 + wx * p11;
                
                // Interpolate along y-axis
                float interp_val = (1.0f - wy) * interp_x0 + wy * interp_x1;

                // Normalize and store
                out_data[(y * out_width + x) * out_channels + c] =
                    interp_val / 255.0f;
            }
        }
    }
}

unsigned char* ResizeImage(const unsigned char* input_data, int input_width,
                           int input_height, int input_channels,
                           int output_width, int output_height) {
  size_t output_size =
      static_cast<size_t>(output_width) * output_height * input_channels;
  unsigned char* output_data = new (std::nothrow) unsigned char[output_size];
  if (!output_data) {
    std::cerr << "Failed to allocate memory for resized image." << std::endl;
    return nullptr;
  }

  int result = stbir_resize_uint8(
      input_data, input_width, input_height, 0, output_data, output_width,
      output_height, 0, input_channels);

  if (!result) {
    std::cerr << "Failed to resize image." << std::endl;
    delete[] output_data;
    return nullptr;
  }

  return output_data;
}

}  // namespace ImageUtils