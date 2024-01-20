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
uint WangHash( uint s ) 
{ 
	s = (s ^ 61) ^ (s >> 16);
	s *= 9, s = s ^ (s >> 4);
	s *= 0x27d4eb2d;
	s = s ^ (s >> 15); 
	return s; 
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
    //In
    __global struct Ray *rays, 
    __global struct Hit *hits,
    __global uint *seeds,
    //Out
    __global struct Ray* newRays,
    __global volatile uint *newRayCounter,
    __global float *accumulators
) {
    int threadIdx = get_global_id(0);

    uint seed = seeds[threadIdx];


    struct Ray ray = rays[threadIdx];
    struct Hit hit = hits[threadIdx];

    // accumulators[3*ray.startThreadId] = 0.0f;
    // accumulators[3*ray.startThreadId+1] = 0.0f;
    // accumulators[3*ray.startThreadId+2] = 1.0f;

    // return;

    if (hit.type != HIT_NOHIT) {
        if (hit.material.type == MAT_LIGHT) {
            accumulators[3*ray.startThreadId] *= hit.material.albedoX;
            accumulators[3*ray.startThreadId+1] *= hit.material.albedoY;
            accumulators[3*ray.startThreadId+2] *= hit.material.albedoZ;
            return;
        }

        //TODO: Add randomness
        float3 hit_normal = (float3)(hit.normalX, hit.normalY, hit.normalZ);
        float3 N = normalize(hit_normal);
        uint seed =  seeds[threadIdx];
        seeds[threadIdx] = seed;
        float3 newRayD = UniformSampleHemisphere(N, &seed);
        struct Ray newRay;
        SetRayD(&newRay, newRayD);
        SetRayO(&newRay, RayO(&ray) + ray.t * RayD(&ray));
        float3 brdf = MaterialAlbedo(&hit.material) * M_1_PI_F;
        float3 irradiance = M_PI_F * 2.0f * brdf * dot(N, newRayD);
        accumulators[3*ray.startThreadId] *= irradiance.x;
        accumulators[3*ray.startThreadId+1] *= irradiance.y;
        accumulators[3*ray.startThreadId+2] *= irradiance.z;
        // accumulators[3*ray.startThreadId] = irradiance.x;
        // accumulators[3*ray.startThreadId+1] = irradiance.y;
        // accumulators[3*ray.startThreadId+2] = irradiance.z;

        // Send extension ray
        newRays[atomic_inc(newRayCounter)] = newRay;
    } else {
            accumulators[3*ray.startThreadId] = 0.0f;
            accumulators[3*ray.startThreadId+1] = 0.0f;
            accumulators[3*ray.startThreadId+2] = 0.0f;
    }


    //return;

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