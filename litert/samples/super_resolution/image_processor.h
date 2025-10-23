#ifndef THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_SUPER_RES_IMAGE_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_SUPER_RES_IMAGE_PROCESSOR_H_

#include <stdbool.h>

#include <cstddef>
#include <string>

// EGL
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

class ImageProcessor {
   public:
    ImageProcessor();
    ~ImageProcessor();

    bool InitializeGL(const std::string& passthrough_vert_shader_path, const std::string& super_res_compute_shader_path);
    void ShutdownGL();

    GLuint CreateOpenGLTexture(const unsigned char* image_data, int width, int height, int channels);
    void DeleteOpenGLTexture(GLuint texture_id);
    GLuint CreateOpenGLBuffer(const void* data, size_t data_size, GLenum usage = GL_STATIC_DRAW);
    void DeleteOpenGLBuffer(GLuint buffer_id);

    bool PreprocessInputForSuperResolution(GLuint input_tex_id, int output_width, int output_height, GLuint preprocessed_buffer_id);

    bool ReadBufferData(GLuint buffer_id, size_t offset, size_t data_size, void* out_data);

   private:
    EGLDisplay egl_display_ = EGL_NO_DISPLAY;
    EGLSurface egl_surface_ = EGL_NO_SURFACE;
    EGLContext egl_context_ = EGL_NO_CONTEXT;

    GLuint super_res_compute_shader_program_ = 0;
    GLuint fbo_ = 0;

    void CleanupEGLContext();
    void CleanupEGLSurface();
    void CleanupGLResources();

    bool SetupComputeShader(const std::string& compute_shader_path, GLuint& program_id);
};

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_SAMPLES_SUPER_RES_IMAGE_PROCESSOR_H_