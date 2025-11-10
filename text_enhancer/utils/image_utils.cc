#include <algorithm>  // For std::min/std::max
#include <cmath>      // For std::floor
#include <new> // For std::nothrow
#include <iostream>
#include <cstring> // For std::memcpy
#include <cstdint> // <-- NEWLY ADDED

#include "image_utils.h"

// --- NEW: Include AHB header for Android ---
#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#endif
// -----------------------------------------

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

unsigned char* CropImage(const unsigned char* in_data,
                         int in_width, int in_height, int in_channels,
                         int crop_x, int crop_y,
                         int crop_width, int crop_height) {
  if (!in_data) {
    std::cerr << "Input image data is null." << std::endl;
    return nullptr;
  }

  // Check if crop dimensions are valid
  if (crop_x < 0 || crop_y < 0 ||
      (crop_x + crop_width) > in_width ||
      (crop_y + crop_height) > in_height) {
    std::cerr << "Crop dimensions are outside the image bounds." << std::endl;
    return nullptr;
  }

  // Allocate memory for the new cropped image
  size_t out_size =
      static_cast<size_t>(crop_width) * crop_height * in_channels;
  unsigned char* out_data = new (std::nothrow) unsigned char[out_size];
  if (!out_data) {
    std::cerr << "Failed to allocate memory for cropped image." << std::endl;
    return nullptr;
  }

  const size_t in_row_stride_bytes = in_width * in_channels;
  const size_t out_row_stride_bytes = crop_width * in_channels; // This is also the amount to copy

  // Point to the start of the first row in the crop region
  const unsigned char* in_ptr =
      in_data + (crop_y * in_row_stride_bytes) + (crop_x * in_channels);
  
  unsigned char* out_ptr = out_data;

  // Copy each row of the crop region
  for (int y = 0; y < crop_height; ++y) {
    std::memcpy(out_ptr, in_ptr, out_row_stride_bytes);
    
    // Move pointers to the next row
    in_ptr += in_row_stride_bytes;
    out_ptr += out_row_stride_bytes;
  }

  return out_data;
}


// --- NEW: Add AHB utility functions for Android ---
#ifdef __ANDROID__
AHardwareBuffer* CreateAhbFromImageData(const unsigned char* data, int width, int height) {
    AHardwareBuffer* ahb = nullptr;
    AHardwareBuffer_Desc ahb_desc = {};
    ahb_desc.width = width;
    ahb_desc.height = height;
    ahb_desc.layers = 1;
    // Format must match what the loader gives (RGBA8) and what Vulkan expects
    ahb_desc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM; 
    // Usage: CPU can write, GPU can sample (read)
    ahb_desc.usage = AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE;
    
    // --- FIX: Explicitly request a stride equal to the width ---
    // This is a request, not a guarantee.
    ahb_desc.stride = width;
    // ---------------------------------------------------------

    if (AHardwareBuffer_allocate(&ahb_desc, &ahb) != 0) {
        std::cerr << "Failed to allocate AHardwareBuffer." << std::endl;
        return nullptr;
    }

    // --- FIX: Get the *actual* description from the allocated buffer ---
    // The driver may have given us a different stride than requested.
    AHardwareBuffer_Desc actual_desc = {};
    AHardwareBuffer_describe(ahb, &actual_desc);
    // -----------------------------------------------------------------

    void* ahb_data = nullptr;
    // Lock the buffer for writing
    if (AHardwareBuffer_lock(ahb, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, &ahb_data) != 0) {
        std::cerr << "Failed to lock AHardwareBuffer for writing." << std::endl;
        AHardwareBuffer_release(ahb);
        return nullptr;
    }

    // --- FIX: Perform a row-by-row copy respecting the buffer's stride ---
    const uint32_t bytes_per_pixel = 4; // RGBA8
    const uint32_t src_stride_bytes = width * bytes_per_pixel;
    
    // Use the *actual* stride from AHardwareBuffer_describe
    const uint32_t dst_stride_bytes = actual_desc.stride * bytes_per_pixel; 

    auto src_ptr = static_cast<const uint8_t*>(data);
    auto dst_ptr = static_cast<uint8_t*>(ahb_data);

    if (src_stride_bytes == dst_stride_bytes) {
        // Optimization: If strides match, we can do a single fast copy.
        memcpy(dst_ptr, src_ptr, height * src_stride_bytes);
    } else {
        // Otherwise, copy row by row
        std::cout << "[Debug CreateAhb] Warning: AHB stride (" << actual_desc.stride
                  << ") != width (" << width << "). Performing row-by-row copy." << std::endl;
        for (int y = 0; y < height; ++y) {
            memcpy(dst_ptr, src_ptr, src_stride_bytes); // Copy one row's worth of data (width * 4)
            dst_ptr += dst_stride_bytes; // Go to next row in destination (using actual stride)
            src_ptr += src_stride_bytes; // Go to next row in source
        }
    }
    // -------------------------------------------------------------------

    // Unlock the buffer
    AHardwareBuffer_unlock(ahb, nullptr);

    std::cout << "Successfully created and populated AHardwareBuffer (width: " << actual_desc.width
              << ", stride: " << actual_desc.stride << ")." << std::endl;
    return ahb;
}

void FreeAhb(AHardwareBuffer* buffer) {
    if (buffer) {
        AHardwareBuffer_release(buffer);
    }
}
#endif // __ANDROID__
// ------------------------------------------------

}  // namespace ImageUtils