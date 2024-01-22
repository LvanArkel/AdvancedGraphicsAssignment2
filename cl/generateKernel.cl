#include "cl/wavefrontTools.cl"

__kernel void generate(
    // In
    const int image_width, const int image_height,
    float3 camPos, float3 p0, float3 p1, float3 p2,
    // Out
    __global struct Ray* rays,
    __global float* Ts,
    __global float* Es
) {
    int threadIdx = get_global_id(0);

    int x = threadIdx % image_width;
    int y = (threadIdx / image_width) % (image_height);
    
	float3 pixelPos = p0 +
		(p1 - p0) * ((float)x / image_width) +
		(p2 - p0) * ((float)y / image_height);

    struct Ray ray;
    SetRayO(&ray, camPos);
    SetRayD(&ray, normalize(pixelPos - camPos));
    ray.t = 1e30f;
    ray.startThreadId = threadIdx;

    rays[threadIdx] = ray;

    for (int i = 0; i < 3; i++) {
        Ts[3*threadIdx+i] = 1.0f;
        Es[3*threadIdx+i] = 0.0f;
    }
}