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
    __global volatile uint *rayCount,
    __global volatile uint *newRayCounter,
    __global float4 *accumulators
) {
    int rayIdx = atomic_dec(rayCount) - 1;
    if (rayIdx < 0) {
        atomic_max(rayCount, 0);
        return;
    }

    struct Ray ray = rays[rayIdx];
    struct Hit hit = hits[rayIdx];

    if (hit.type != HIT_NOHIT) {
        if (hit.material.type == MAT_LIGHT) {
            accumulators[ray.startThreadId].x *= hit.material.albedoX;
            accumulators[ray.startThreadId].y *= hit.material.albedoY;
            accumulators[ray.startThreadId].z *= hit.material.albedoZ;
            accumulators[ray.startThreadId].a = 1.0f;
            return;
        }

        float3 hit_normal = (float3)(hit.normalX, hit.normalY, hit.normalZ);
        float3 N = normalize(hit_normal);
        uint seed =  seeds[rayIdx];

        float3 newRayD = UniformSampleHemisphere(N, &seed);
        struct Ray newRay;
        SetRayD(&newRay, newRayD);
        SetRayO(&newRay, RayO(&ray) + ray.t * RayD(&ray));
        newRay.t = 1e30f;
        newRay.startThreadId = ray.startThreadId;
        float3 brdf = MaterialAlbedo(&hit.material) * M_1_PI_F;
        float3 irradiance = M_PI_F * 2.0f * brdf * dot(N, newRayD);
        accumulators[ray.startThreadId].x *= irradiance.x;
        accumulators[ray.startThreadId].y *= irradiance.y;
        accumulators[ray.startThreadId].z *= irradiance.z;

        // Send extension ray
        newRays[atomic_inc(newRayCounter)] = newRay;
        seeds[rayIdx] = seed;
    } else {
        accumulators[ray.startThreadId] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
}