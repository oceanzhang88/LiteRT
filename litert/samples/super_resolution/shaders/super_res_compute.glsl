#version 310 es
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Input texture (original image)
layout(binding = 0) uniform sampler2D inputTexture;

// Output buffer (resized and normalized image for the model)
layout(std430, binding = 1) buffer OutputBuffer {
    float data[];
} output_buffer;

// Uniforms to pass dimensions from C++
uniform ivec2 output_dims;

void main() {
    ivec2 store_pos = ivec2(gl_GlobalInvocationID.xy);
    int out_width = output_dims.x;
    int out_height = output_dims.y;

    if (store_pos.x >= out_width || store_pos.y >= out_height) {
        return;
    }

    // Calculate normalized texture coordinates
    vec2 tex_coord = vec2(float(store_pos.x) / float(out_width - 1),
                          float(store_pos.y) / float(out_height - 1));

    // Sample the input texture
    vec3 pixel = texture(inputTexture, tex_coord).rgb;

    // Write to the output buffer (assuming RGB format)
    int index = (store_pos.y * out_width + store_pos.x) * 3;
    output_buffer.data[index] = pixel.r;
    output_buffer.data[index + 1] = pixel.g;
    output_buffer.data[index + 2] = pixel.b;
}