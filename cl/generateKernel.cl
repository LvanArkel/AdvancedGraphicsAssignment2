#include "cl/wavefrontTools.cl"

__kernel void generate(
    const int image_width, const int image_height,
    float3 camPos, float3 p0, float3 p1, float3 p2,
    __global struct Ray* rays,
    __global float* accumulators
) {
    int threadIdx = get_global_id(0);

    int x = threadIdx % image_width;
    int y = (threadIdx / image_width) % (image_height);

    int pixelIdx = threadIdx % (image_width * image_height);
    
	float3 pixelPos = p0 +
		(p1 - p0) * ((float)x / image_width) +
		(p2 - p0) * ((float)y / image_height);

    struct Ray ray;
    SetRayO(&ray, camPos);
    SetRayD(&ray, normalize(pixelPos - camPos));
    ray.t = 1e30f;
    ray.pixelIdx = pixelIdx;

    rays[threadIdx] = ray;

    accumulators[3*pixelIdx] = 0.0f;
    accumulators[3*pixelIdx+1] = 0.0f;
    accumulators[3*pixelIdx+2] = 0.0f;
}