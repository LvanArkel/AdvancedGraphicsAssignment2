#include "cl/wavefrontTools.cl"

__kernel void finalize(
    __global int* pixels,
    __global float* Es,
    const int image_size,
    const int samples
) {
    int threadIdx = get_global_id(0);

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    for (int i = 0; i < samples; i++) {
        r += Es[3*(threadIdx+image_size*i)];
        g += Es[3*(threadIdx+image_size*i)+1];
        b += Es[3*(threadIdx+image_size*i)+2];
    }
    float invSamples = 1 / (float)samples;
    r *= invSamples;
    g *= invSamples;
    b *= invSamples;

    int ri = (int)(min(1.0f, r) * 255.0);
    int gi = (int)(min(1.0f, g) * 255.0);
    int bi = (int)(min(1.0f, b) * 255.0);

    pixels[threadIdx] = ri << 16 | gi << 8 | bi;
}