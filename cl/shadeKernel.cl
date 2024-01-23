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

struct LightCheck {
    int sphereIndex;
    float3 L;
    float3 Nl;
    float dist;
    float A;
};

float3 UniformSampleSphere(uint* seed) {
    float3 result;
	do {
		result = (float3)(RandomFloat(seed)*2.0f - 1.0f, RandomFloat(seed) * 2.0f - 1.0f, RandomFloat(seed) * 2.0f - 1.0f);
	} while (length(result) > 1);
    return normalize(result);
}

struct LightCheck SampleRandomLight(
    uint* seed,
    int lightSize,
    int* lightIndices,
    struct Sphere* spheres,
    float3 I
) {
    int selectedLight;
    do {
        selectedLight = (int) (RandomFloat(seed) * lightSize);
    } while (selectedLight == lightSize);
    struct LightCheck lightCheck;
    struct Sphere selectedSphere = spheres[lightIndices[selectedLight]];
    float3 origin = SphereOrigin(&selectedSphere);
    float3 Nl = UniformSampleSphere(seed);
    float3 P = selectedSphere.radius * Nl + origin;
    float3 intersectDirection = P - I;
    lightCheck.sphereIndex = lightIndices[selectedLight];
    lightCheck.L = normalize(intersectDirection);
    lightCheck.Nl = Nl;
    lightCheck.dist = length(intersectDirection);
    lightCheck.A = 4.0f * M_PI_F * selectedSphere.radius * selectedSphere.radius;

    return lightCheck;
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
    __global struct Sphere* spheres,
    __global struct Material* sphereMaterials,
    __global int* lightSpheres,
    const int lightSphereSize,
    //Out
    __global struct Ray* newRays,
    __global volatile uint *newRayCounter,
    __global struct ShadowRay* shadowRays,
    __global volatile uint *shadowRayCounter,
    __global float* Ts
) {
    int threadIdx = get_global_id(0);

    uint seed = seeds[threadIdx];


    struct Ray ray = rays[threadIdx];
    struct Hit hit = hits[threadIdx];

    if (hit.type != HIT_NOHIT) {
        if (hit.material.type == MAT_LIGHT) {
            return;
        }

        float3 I = RayO(&ray) + ray.t * RayD(&ray);
        float3 N = normalize(HitNormal(&hit));
        float3 brdf = MaterialAlbedo(&hit.material) * M_1_PI_F;
        // Sample random light
        struct LightCheck lightCheck = SampleRandomLight(&seed, lightSphereSize, lightSpheres, spheres, I);
        if (dot(N, lightCheck.L) > 0 && dot( lightCheck.Nl, -lightCheck.L) > 0) {
            float solidAngle = (dot(lightCheck.Nl, -lightCheck.L) * lightCheck.A) / (lightCheck.dist * lightCheck.dist);
            float lightPdf = 1.0f / solidAngle;
            float3 T = (float)(Ts[3*ray.startThreadId], Ts[3*ray.startThreadId+1], Ts[3*ray.startThreadId+2]);
            float3 deltaE = T * (dot (N, lightCheck.L) / lightPdf) * brdf * MaterialAlbedo(&sphereMaterials[lightCheck.sphereIndex]);
            struct ShadowRay shadowRay;
            SetRayO(&shadowRay.ray, I);
            SetRayD(&shadowRay.ray, lightCheck.L);
            shadowRay.ray.t = lightCheck.dist;
            shadowRay.ray.startThreadId = ray.startThreadId;
            SetShadowRayDE(&shadowRay, deltaE);
            shadowRays[atomic_inc(shadowRayCounter)] = shadowRay;
        }
        

        // Continue random walk
        float3 newRayD = UniformSampleHemisphere(N, &seed);
        struct Ray newRay;
        SetRayD(&newRay, newRayD);
        SetRayO(&newRay, I);
        newRay.t = 1e30f;
        newRay.startThreadId = ray.startThreadId;
        float3 irradiance = M_PI_F * 2.0f * brdf * dot(N, newRayD);
        for (int i = 0; i < 3; i++) {
            Ts[3*ray.startThreadId+i] = irradiance[i];
        }
        newRays[atomic_inc(newRayCounter)] = newRay;
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