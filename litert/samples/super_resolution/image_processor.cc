#include "litert/samples/super_resolution/image_processor.h"

#include <GLES3/gl32.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "absl/log/absl_check.h"

namespace {

// Helper functions for error checking (GL and EGL)
void GlCheckErrorDetail(const char* file, int line, const char* operation = "") {
    GLenum error_code;
    while ((error_code = glGetError()) != GL_NO_ERROR) {
        std::string error_string;
        switch (error_code) {
            case GL_INVALID_ENUM:
                error_string = "INVALID_ENUM";
                break;
            case GL_INVALID_VALUE:
                error_string = "INVALID_VALUE";
                break;
            case GL_INVALID_OPERATION:
                error_string = "INVALID_OPERATION";
                break;
            case GL_OUT_OF_MEMORY:
                error_string = "OUT_OF_MEMORY";
                break;
            case GL_INVALID_FRAMEBUFFER_OPERATION:
                error_string = "INVALID_FRAMEBUFFER_OPERATION";
                break;
            default:
                error_string = "UNKNOWN_ERROR_CODE_0x" + std::to_string(error_code);
                break;
        }
        std::cerr << "GL_ERROR (" << operation << "): " << error_string << " - " << file << ":" << line << std::endl;
    }
}
#define GlCheckErrorOp(op) GlCheckErrorDetail(__FILE__, __LINE__, op)

void EglCheckErrorDetail(const char* file, int line, const char* operation = "") {
    EGLint egl_error_code = eglGetError();
    if (egl_error_code != EGL_SUCCESS) {
        std::cerr << "EGL_ERROR (" << operation << "): code 0x" << std::hex << egl_error_code << " at " << file << ":" << line << std::endl;
    }
}
#define EglCheckErrorOp(op) EglCheckErrorDetail(__FILE__, __LINE__, op)

// Helper function to load shader source from a file
std::string LoadShaderSourceFromFile(const std::string& filepath) {
    std::ifstream file_stream(filepath);
    if (!file_stream.is_open()) {
        ABSL_CHECK(false) << "Failed to open shader file: " + filepath;
    }
    std::stringstream buffer;
    buffer << file_stream.rdbuf();
    return buffer.str();
}

// Helper function to compile a shader
GLuint CompileShaderInternal(GLenum type, const std::string& source_code) {
    GLuint shader_id = glCreateShader(type);
    const char* source_code_ptr = source_code.c_str();
    glShaderSource(shader_id, 1, &source_code_ptr, nullptr);
    glCompileShader(shader_id);
    GLint success_status;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success_status);
    if (!success_status) {
        GLchar info_log[1024];
        glGetShaderInfoLog(shader_id, sizeof(info_log), nullptr, info_log);
        std::string shader_type_string = (type == GL_COMPUTE_SHADER) ? "COMPUTE" : "UNKNOWN";
        glDeleteShader(shader_id);
        ABSL_CHECK(false) << "SHADER " + shader_type_string + " compilation failed:\n" + info_log;
    }
    return shader_id;
}

}  // anonymous namespace

ImageProcessor::ImageProcessor() = default;

ImageProcessor::~ImageProcessor() { ShutdownGL(); }

bool ImageProcessor::SetupComputeShader(const std::string& compute_shader_path, GLuint& program_id) {
    std::cout << "Setting up compute shader: " << compute_shader_path << std::endl;
    std::string compute_source = LoadShaderSourceFromFile(compute_shader_path);
    GLuint compute_shader_id = CompileShaderInternal(GL_COMPUTE_SHADER, compute_source);

    program_id = glCreateProgram();
    glAttachShader(program_id, compute_shader_id);
    glLinkProgram(program_id);

    GLint link_success_status;
    glGetProgramiv(program_id, GL_LINK_STATUS, &link_success_status);
    if (!link_success_status) {
        GLchar info_log[1024];
        glGetProgramInfoLog(program_id, sizeof(info_log), nullptr, info_log);
        ABSL_CHECK(false) << "Compute shader program linking failed for " + compute_shader_path + ":\n" + info_log;
    }

    glDeleteShader(compute_shader_id);
    return program_id != 0;
}

bool ImageProcessor::InitializeGL(const std::string& passthrough_vert_shader_path, const std::string& super_res_compute_shader_path) {
    egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_display_ == EGL_NO_DISPLAY) {
        std::cerr << "Failed to get EGL display." << std::endl;
        return false;
    }
    if (!eglInitialize(egl_display_, nullptr, nullptr)) {
        std::cerr << "Failed to initialize EGL." << std::endl;
        return false;
    }

    const EGLint config_attribs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_NONE};
    EGLConfig config;
    EGLint num_configs;
    if (!eglChooseConfig(egl_display_, config_attribs, &config, 1, &num_configs) || num_configs == 0) {
        std::cerr << "Failed to choose EGL config." << std::endl;
        return false;
    }

    const EGLint pbuffer_attribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    egl_surface_ = eglCreatePbufferSurface(egl_display_, config, pbuffer_attribs);
    if (egl_surface_ == EGL_NO_SURFACE) {
        std::cerr << "Failed to create EGL pbuffer surface." << std::endl;
        return false;
    }

    const EGLint context_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE};
    egl_context_ = eglCreateContext(egl_display_, config, EGL_NO_CONTEXT, context_attribs);
    if (egl_context_ == EGL_NO_CONTEXT) {
        std::cerr << "Failed to create EGL context." << std::endl;
        return false;
    }

    if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
        std::cerr << "Failed to make EGL context current." << std::endl;
        return false;
    }

    if (!SetupComputeShader(super_res_compute_shader_path, super_res_compute_shader_program_)) {
        std::cerr << "Failed to setup super resolution compute shader." << std::endl;
        ShutdownGL();
        return false;
    }

    glGenFramebuffers(1, &fbo_);
    return true;
}

void ImageProcessor::ShutdownGL() {
    CleanupGLResources();
    if (egl_display_ != EGL_NO_DISPLAY) {
        eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        CleanupEGLContext();
        CleanupEGLSurface();
        eglTerminate(egl_display_);
        egl_display_ = EGL_NO_DISPLAY;
    }
}

GLuint ImageProcessor::CreateOpenGLTexture(const unsigned char* image_data, int width, int height, int channels) {
    if (!image_data || width <= 0 || height <= 0) return 0;

    GLenum format = (channels == 3) ? GL_RGB : GL_RGBA;
    GLenum internal_format = (channels == 3) ? GL_RGB8 : GL_RGBA8;

    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, format, GL_UNSIGNED_BYTE, image_data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture_id;
}

void ImageProcessor::DeleteOpenGLTexture(GLuint texture_id) {
    if (texture_id != 0) {
        glDeleteTextures(1, &texture_id);
    }
}

GLuint ImageProcessor::CreateOpenGLBuffer(const void* data, size_t data_size, GLenum usage) {
    if (data_size <= 0) return 0;

    GLuint buffer_id;
    glGenBuffers(1, &buffer_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, data_size, data, usage);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return buffer_id;
}

void ImageProcessor::DeleteOpenGLBuffer(GLuint buffer_id) {
    if (buffer_id != 0) {
        glDeleteBuffers(1, &buffer_id);
    }
}

bool ImageProcessor::PreprocessInputForSuperResolution(GLuint input_tex_id, int output_width, int output_height, GLuint preprocessed_buffer_id) {
    if (input_tex_id == 0 || preprocessed_buffer_id == 0) return false;
    if (super_res_compute_shader_program_ == 0) return false;

    glUseProgram(super_res_compute_shader_program_);

    // Set the output dimensions uniform
    glUniform2i(glGetUniformLocation(super_res_compute_shader_program_, "output_dims"), output_width, output_height);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, input_tex_id);
    glUniform1i(glGetUniformLocation(super_res_compute_shader_program_, "inputTexture"), 0);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, preprocessed_buffer_id);

    glDispatchCompute((output_width + 7) / 8, (output_height + 7) / 8, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glFinish();

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    return true;
}

bool ImageProcessor::ReadBufferData(GLuint buffer_id, size_t offset, size_t data_size, void* out_data) {
    if (buffer_id == 0 || data_size == 0 || out_data == nullptr) return false;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id);
    void* mapped_ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, offset, data_size, GL_MAP_READ_BIT);
    if (!mapped_ptr) {
        GlCheckErrorOp("glMapBufferRange");
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        return false;
    }

    memcpy(out_data, mapped_ptr, data_size);

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return true;
}

void ImageProcessor::CleanupEGLContext() {
    if (egl_display_ != EGL_NO_DISPLAY && egl_context_ != EGL_NO_CONTEXT) {
        eglDestroyContext(egl_display_, egl_context_);
        egl_context_ = EGL_NO_CONTEXT;
    }
}

void ImageProcessor::CleanupEGLSurface() {
    if (egl_display_ != EGL_NO_DISPLAY && egl_surface_ != EGL_NO_SURFACE) {
        eglDestroySurface(egl_display_, egl_surface_);
        egl_surface_ = EGL_NO_SURFACE;
    }
}

void ImageProcessor::CleanupGLResources() {
    if (fbo_) {
        glDeleteFramebuffers(1, &fbo_);
        fbo_ = 0;
    }
    if (super_res_compute_shader_program_) {
        glDeleteProgram(super_res_compute_shader_program_);
        super_res_compute_shader_program_ = 0;
    }
}