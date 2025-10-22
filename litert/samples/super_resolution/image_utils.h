#ifndef THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_SUPER_RESOLUTION_IMAGE_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_SUPER_RESOLUTION_IMAGE_UTILS_H_

#include <string>

namespace ImageUtils {

unsigned char* LoadImage(const std::string& file_path, int& width, int& height,
                         int& channels_in_file, int desired_channels);
void FreeImageData(unsigned char* data);
bool SaveImage(const std::string& file_path, int width, int height,
               int channels, const unsigned char* data);

}  // namespace ImageUtils

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_SUPER_RESOLUTION_IMAGE_UTILS_H_