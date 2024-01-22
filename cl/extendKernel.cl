#include "cl/wavefrontTools.cl"

struct Hit Trace(
    struct Ray* ray,
    __global struct Tri* tri, const int NTris,
    __global struct Sphere* spheres, const int NSpheres,
    __global struct Material* triangleMaterials, __global struct Material* sphereMaterials
) {
	int lastIntersect = -1;
	bool hitSphere = false;

	for (int i = 0; i < NTris; i++) {
		float lastT = ray->t;
		IntersectTri(ray, &tri[i]);
		if (lastT != ray->t) {
			lastIntersect = i;
		}
	}
    
	for (int i = 0; i < NSpheres; i++) {
		float lastT = ray->t;
		IntersectSphere(ray, &spheres[i]);
		if (lastT != ray->t) {
			lastIntersect = i;
			hitSphere = true;
		}
	}

	struct Hit hit;
	if (lastIntersect != -1) {
		if (hitSphere) {
			hit.type = HIT_SPHERE;
			hit.index = lastIntersect;
			struct Sphere sphere = spheres[lastIntersect];
			hit.material = sphereMaterials[lastIntersect];
            float3 rayO = RayO(ray);
            float3 rayD = RayD(ray);
			float3 normal = (rayO + ray->t * rayD) - SphereOrigin(&sphere);
            SetHitNormal(&hit, normal);
		}
		else {
			hit.type = HIT_TRIANGLE;
			hit.index = lastIntersect;

			struct Tri triangle = tri[lastIntersect];
			float3 vertex0 = TriVertex0(&triangle);
			float3 vertex1 = TriVertex1(&triangle);
			float3 vertex2 = TriVertex2(&triangle);

			hit.material = triangleMaterials[lastIntersect];
			float3 normal = cross(vertex1 - vertex0, vertex2 - vertex0);
            SetHitNormal(&hit, normal);
		}
	}
	else {
		hit.type = HIT_NOHIT;
	}
	return hit;
}

__kernel void extend(
    __global struct Tri* tris, const int NTris,
    __global struct Sphere* spheres, const int NSpheres,
    __global struct Material* triangleMaterials, __global struct Material* sphereMaterials,
    __global struct Ray* rays,
    __global struct Hit* hits
) {
    int threadIdx = get_global_id(0);
    struct Ray* ray = &rays[threadIdx];

    hits[threadIdx] = Trace(
        ray,
        tris, NTris,
        spheres, NSpheres,
        triangleMaterials, sphereMaterials
    );
}