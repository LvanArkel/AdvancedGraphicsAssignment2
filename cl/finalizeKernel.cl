#include "cl/wavefrontTools.cl"

__kernel void finalize(
    __global int* pixels,
    __global float4* accumulators,
    const int image_size,
    const int samples,
    __global float4* clr
) {
    int threadIdx = get_global_id(0);

    if (accumulators[threadIdx].a) {
        clr[threadIdx].x += accumulators[threadIdx].x;
        clr[threadIdx].y += accumulators[threadIdx].y;
        clr[threadIdx].z += accumulators[threadIdx].z;
    }

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float3 rgb = (float3)(0.0f, 0.0f, 0.0f);
    rgb = (float3)(clr[threadIdx].x, clr[threadIdx].y, clr[threadIdx].z);

    rgb/=(float)samples;

    int ri = (int)(min(1.0f, rgb.x) * 255.0);
    int gi = (int)(min(1.0f, rgb.y) * 255.0);
    int bi = (int)(min(1.0f, rgb.z) * 255.0);

    pixels[threadIdx] = ri << 16 | gi << 8 | bi;

    clr[threadIdx].x = 0.0f;
    clr[threadIdx].y = 0.0f;
    clr[threadIdx].z = 0.0f;
    clr[threadIdx].a = 0.0f;
}