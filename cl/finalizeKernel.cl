#include "cl/wavefrontTools.cl"

__kernel void finalize(
    __global int* pixels,
    __global float4* accumulators,
    const int image_size,
    const int samples
) {
    int threadIdx = get_global_id(0);

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float3 rgb = (float3)(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < samples; i++) {
        if (accumulators[threadIdx + image_size * i].a == 1.0f) {
            rgb += accumulators[threadIdx + image_size * i].xyz;
        }
        // r += accumulators[3*(threadIdx + image_size*i)];
        // g += accumulators[3*(threadIdx + image_size*i)+1];
        // b += accumulators[3*(threadIdx + image_size*i)+2];
    }
    float invSamples = 1 / (float)samples;
    rgb *= invSamples;
    // r *= invSamples;
    // g *= invSamples;
    // b *= invSamples;

    int ri = (int)(min(1.0f, rgb.x) * 255.0);
    int gi = (int)(min(1.0f, rgb.y) * 255.0);
    int bi = (int)(min(1.0f, rgb.z) * 255.0);
    // int ri = (int)(min(1.0f, r) * 255.0);
    // int gi = (int)(min(1.0f, g) * 255.0);
    // int bi = (int)(min(1.0f, b) * 255.0);

    pixels[threadIdx] = ri << 16 | gi << 8 | bi;
}