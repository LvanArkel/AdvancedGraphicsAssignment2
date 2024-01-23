#include "cl/wavefrontTools.cl"

__kernel void generate(
    const int image_width, const int image_height,
    float3 camPos, float3 p0, float3 p1, float3 p2,
    __global struct Ray* rays,
    __global float4* accumulators,
    __global float4* clr
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

    clr[threadIdx].x += accumulators[threadIdx].x;
    clr[threadIdx].y += accumulators[threadIdx].y;
    clr[threadIdx].z += accumulators[threadIdx].z;
    accumulators[threadIdx].x = 1.0f;// = (float4)(1.0f, 1.0f, 1.0f, 0.0f);
    accumulators[threadIdx].y = 1.0f;
    accumulators[threadIdx].z = 1.0f;//

    // accumulators[3*threadIdx] = 1.0f;
    // accumulators[3*threadIdx+1] = 1.0f;
    // accumulators[3*threadIdx+2] = 1.0f;
}