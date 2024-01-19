#include "cl/wavefrontTools.cl"

__kernel void finalize(
    __global int* pixels,
    __global float* accumulators,
    const float inverseSamples
) {
    int threadIdx = get_global_id(0);

    float r = accumulators[3*threadIdx] * inverseSamples;
    float g = accumulators[3*threadIdx+1] * inverseSamples;
    float b = accumulators[3*threadIdx+2] * inverseSamples;

    struct Hit hit;

    int ri = (int)(min(1.0f, r) * 255.0);
    int gi = (int)(min(1.0f, g) * 255.0);
    int bi = (int)(min(1.0f, b) * 255.0);

    pixels[threadIdx] = ri << 16 | gi << 8 | bi;
}