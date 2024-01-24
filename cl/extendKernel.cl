#include "cl/wavefrontTools.cl"

void IntersectSphere(struct Ray* ray, struct Sphere* sphere) {
    float3 sphere_origin = SphereOrigin(sphere);
    float3 rayO = RayO(ray);
    float3 rayD = RayD(ray);
	float3 oc = rayO - sphere_origin;
	float b = dot(oc, rayD);
	float c = dot(oc, oc) - sphere->radius * sphere->radius;
	float h = b * b - c;
	if (h > 0.0f) {
		h = sqrt(h);
		ray->t = min(ray->t, -b - h);
	}
}

void IntersectTri(struct Ray* ray, struct Tri* tri )
{
    float3 vertex0 = TriVertex0(tri);
    float3 vertex1 = TriVertex1(tri);
    float3 vertex2 = TriVertex2(tri);
    float3 rayO = RayO(ray);
    float3 rayD = RayD(ray);
	float3 edge1 = vertex1 - vertex0;
	float3 edge2 = vertex2 - vertex0;
	float3 h = cross( rayD, edge2 );
	float a = dot( edge1, h );
	if (a > -0.0001f && a < 0.0001f) return; // ray parallel to triangle
	float f = 1 / a;
	float3 s = rayO - vertex0;
	float u = f * dot( s, h );
	if (u < 0 || u > 1) return;
	float3 q = cross( s, edge1 );
	float v = f * dot( rayD, q );
	if (v < 0 || u + v > 1) return;
	float t = f * dot( edge2, q );
	if (t > 0.0001f) ray->t = min( ray->t, t );
}

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
			float3 normal = SphereOrigin(&sphere) - (rayO + ray->t * rayD);
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
    __global struct Hit* hits,
	__global volatile int *rayCount,
	__global volatile int *newRayCount
) {
	if (*rayCount == 0 && *newRayCount == 0) {
		return;
	}

	if (atomic_cmpxchg(rayCount, 0, *newRayCount) == 0) {
		*newRayCount = 0;
	}

    int threadIdx = get_global_id(0);
	if (threadIdx >= *rayCount) {
		return;
	}

    struct Ray* ray = &rays[threadIdx];

    hits[threadIdx] = Trace(
        ray,
        tris, NTris,
        spheres, NSpheres,
        triangleMaterials, sphereMaterials
    );
}