#include "cl/wavefrontTools.cl"

uint RandomInt( uint* s ) // Marsaglia's XOR32 RNG
{ 
	*s ^= *s << 13;
	*s ^= *s >> 17;
	* s ^= *s << 5; 
	return *s; 
}
float RandomFloat( uint* s ) 
{ 
	return RandomInt( s ) * 2.3283064365387e-10f; // = 1 / (2^32-1)
}

// Returns a normalized vector of which the angle is on the same hemisphere as the normal.
float3 UniformSampleHemisphere(float3 normal, uint* seed) {
	float3 result;
	do {
		result = (float3)(RandomFloat(seed)*2.0f - 1.0f, RandomFloat(seed) * 2.0f - 1.0f, RandomFloat(seed) * 2.0f - 1.0f);
	} while (length(result) > 1);
	if (dot(result, normal) < 0) {
		result = -result;
	}
	// Normalize result
	return normalize(result);
}

__kernel void shade(
    __global struct Ray *rays, 
    __global struct Hit *hits,
    __global float *accumulators
) {
    int threadIdx = get_global_id(0);

    struct Ray ray = rays[threadIdx];
    struct Hit hit = hits[threadIdx];

    float3 color;
    bool bounce;

    if (hit.type == HIT_NOHIT) {
        accumulators[3*ray.pixelIdx] += 0.0f;
        accumulators[3*ray.pixelIdx+1] += 0.0f;
        accumulators[3*ray.pixelIdx+2] += 0.0f;
    } else {
        
        accumulators[3*ray.pixelIdx] += hit.material.albedoX;
        accumulators[3*ray.pixelIdx+1] += hit.material.albedoY;
        accumulators[3*ray.pixelIdx+2] += hit.material.albedoZ;
    }
    return;

    // if (hit.type == HIT_NOHIT) {
    //     accumulators[3*ray.pixelIdx] *= 0;
    //     accumulators[3*ray.pixelIdx+1] *= 0;
    //     accumulators[3*ray.pixelIdx+2] *= 0;
    //     bounce = false;
    // } else if (hit.material.type == MAT_LIGHT) {
    //     accumulators[3*ray.pixelIdx] *= hit.material.albedoX;
    //     accumulators[3*ray.pixelIdx+1] *= hit.material.albedoY;
    //     accumulators[3*ray.pixelIdx+2] *= hit.material.albedoZ;
    //     bounce = false;
    // } else if (hit.material.type == MAT_DIFFUSE) {
    //     // Generate new ray
    //     float3 normal = (float3)(hit.normalX, hit.normalY, hit.normalZ);
    //     normal = normalize(normal);
    //     //TODO: Add randomness
    //     uint seed = threadIdx;
    //     // Continue in random direction
    //     float3 newDirection = UniformSampleHemisphere(normal, &seed);
    //     struct Ray newRay;
    //     newRay.Ox = ray.Ox + ray.t * ray.Dx;
    //     newRay.Oy = ray.Oy + ray.t * ray.Dy;
    //     newRay.Oz = ray.Oz + ray.t * ray.Dz;
    //     newRay.Dx = newDirection.x;
    //     newRay.Dy = newDirection.y;
    //     newRay.Dz = newDirection.z;
    //     newRay.t = 1e30f;
    //     newRay.pixelIdx = ray.pixelIdx;
    //     newRays[atomic_inc(newRayCounter)] = newRay;
    //     // Update throughput
    //     float brdfX = M_1_PI * hit.material.albedoX;
    //     float brdfY = M_1_PI * hit.material.albedoY;
    //     float brdfZ = M_1_PI * hit.material.albedoZ;

    //     float constFactor = 2.0f * M_PI_F * dot( normal, newDirection);
    //     accumulators[3*ray.pixelIdx] *= constFactor * brdfX;
    //     accumulators[3*ray.pixelIdx+1] *= constFactor * brdfY;
    //     accumulators[3*ray.pixelIdx+2] *= constFactor * brdfZ;
    // }
}