#include "cl/wavefrontTools.cl"

bool Occluded(
    struct Ray* ray,
    __global struct Tri* tri, const int NTris,
    __global struct Sphere* spheres, const int NSpheres
) {
	int lastIntersect = -1;

	for (int i = 0; i < NTris; i++) {
		float lastT = ray->t;
		IntersectTri(ray, &tri[i]);
		if (lastT != ray->t) {
			return true;
		}
	}
    
	for (int i = 0; i < NSpheres; i++) {
		float lastT = ray->t;
		IntersectSphere(ray, &spheres[i]);
		if (lastT != ray->t) {
			return true;
		}
	}

    return false;
}

__kernel void connect(
    // In
    __global struct ShadowRay* shadowRays,
    __global int *shadowRayCount,
    __global struct Tri* tri, const int NTris,
    __global struct Sphere* spheres, const int NSpheres,
    // Out
    __global float *Es
) {
    int threadIdx = get_global_id(0);
    if (threadIdx >= *shadowRayCount) {
        return;
    }

    struct ShadowRay shadowRay = shadowRays[threadIdx];

    if (!Occluded(
        &shadowRay.ray,
        tri, NTris, spheres, NSpheres
    )) {
        Es[3*shadowRay.ray.startThreadId] += shadowRay.DEx;
        Es[3*shadowRay.ray.startThreadId+1] += shadowRay.DEy;
        Es[3*shadowRay.ray.startThreadId+2] += shadowRay.DEz;
    }
}