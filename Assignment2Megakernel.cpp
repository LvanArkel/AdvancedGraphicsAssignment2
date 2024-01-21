#include "precomp.h"
#include "Assignment2Megakernel.h"
#include <iostream>

// THIS SOURCE FILE:
// Code for the article "How to Build a BVH", part 1: basics. Link:
// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics
// This is bare-bones BVH construction and traversal code, running in
// a minimalistic framework.
// Feel free to copy this code to your own framework. Absolutely no
// rights are reserved. No responsibility is accepted either.
// For updates, follow me on twitter: @j_bikker.

TheApp* CreateApp() { return new Assignment2MegakernelApp(); }

#define GPGPU
//#define CPU
//#define ANALYZE_RESULTS

#define N	12 // triangle count
#define NS  3 //sphere count
#define SAMPLES_PER_PIXEL 20

#ifdef ANALYZE_RESULTS
ofstream timefile;
const string filename = "analyze.csv";
uint frameidx = 0;
#endif

static Kernel* kernel = 0; //megakernel
static Buffer* clbuf_r = 0; // buffer for color r channel
static Buffer* clbuf_g = 0; // buffer for color g channel
static Buffer* clbuf_b = 0; // buffer for color b channel
static Buffer* clbuf_spheres = 0; // buffer for spheres
static Buffer* clbuf_tris = 0; // buffer for triangles
static Buffer* clbuf_mat_sphere = 0; // buffer for sphere materials
static Buffer* clbuf_mat_tri = 0; // buffer for sphere materials

//SOA
int cl_r[SCRWIDTH * SCRHEIGHT]; //color r channel sent to kernel
int cl_g[SCRWIDTH * SCRHEIGHT]; //color g channel sent to kernel
int cl_b[SCRWIDTH * SCRHEIGHT]; //color b channel sent to kernel

// forward declarations

// minimal structs
#ifdef CPU
struct Sphere { float3 origin; float radius; };
struct Tri { float3 vertex0, vertex1, vertex2; float3 centroid; };
#endif
#ifdef GPGPU
struct Sphere { float ox, oy, oz; float radius; };
struct Tri {
	float v0x, v0y, v0z;
	float v1x, v1y, v1z;
	float v2x, v2y, v2z;
};
#endif


#ifdef CPU
enum MaterialType {
	DIFFUSE,
	LIGHT
};

struct DiffuseMat {
	MaterialType type;
	union { float3 albedo; float3 emittance; };
};
#endif
#ifdef GPGPU
#define DIFFUSE 0
#define LIGHT 1
struct DiffuseMat {
	int type;
	float albx, alby, albz;
	float emitx, emity, emitz;
};
#endif

__declspec(align(32)) struct BVHNode
{
	float3 aabbMin, aabbMax;
	uint leftFirst, triCount;
	bool isLeaf() { return triCount > 0; }
};
struct Ray { float3 O, D; float t = 1e30f; };

enum HitType {
	NOHIT,
	TRIANGLE,
	SPHERE,
};

struct Hit {
	HitType type;
	int index;
	float3 normal;
	DiffuseMat material;
};

// application data
Tri tri[N];
DiffuseMat diffuseMaterials[N];

Sphere spheres[NS];
DiffuseMat sphereMaterials[NS];

// functions
#ifdef CPU
void IntersectTri(Ray& ray, const Tri& tri)
{
	const float3 edge1 = tri.vertex1 - tri.vertex0;
	const float3 edge2 = tri.vertex2 - tri.vertex0;
	const float3 h = cross(ray.D, edge2);
	const float a = dot(edge1, h);
	if (a > -0.0001f && a < 0.0001f) return; // ray parallel to triangle
	const float f = 1 / a;
	const float3 s = ray.O - tri.vertex0;
	const float u = f * dot(s, h);
	if (u < 0 || u > 1) return;
	const float3 q = cross(s, edge1);
	const float v = f * dot(ray.D, q);
	if (v < 0 || u + v > 1) return;
	const float t = f * dot(edge2, q);
	if (t > 0.0001f) ray.t = min(ray.t, t);
}

void IntersectSphere(Ray& ray, const Sphere& sphere) {
	float3 oc = ray.O - sphere.origin;
	float b = dot(oc, ray.D);
	float c = dot(oc, oc) - sphere.radius * sphere.radius;
	float h = b * b - c;
	if (h > 0.0f) {
		h = sqrt(h);
		ray.t = min(ray.t, -b - h);
	}
}

// Returns a normalized vector of which the angle is on the same hemisphere as the normal.
float3 UniformSampleHemisphere(float3 normal) {
	float3 result;
	do {
		result = float3(RandomFloat() * 2.0f - 0.5f, RandomFloat() * 2.0f - 0.5f, RandomFloat() * 2.0f - 0.5f);
	} while (length(result) > 1);
	if (dot(result, normal) < 0) {
		result = -result;
	}
	// Normalize result
	return normalize(result);
}

// Traces the ray throughout the scene, testing intersections with triangles and spheres.
// Updates the distance of the ray while tracing.
// When the closest intersection is a triangle, returns 0 <= result < N.
// When the closest intersection is a sphere, returns N <= result < N + NS, where result is the index of the sphere offset by N.
// Calculated normals may not be normalized
Hit Trace(Ray& ray) {
	int lastIntersect = -1;
	bool hitSphere = false;
	for (int i = 0; i < N; i++) {
		float lastT = ray.t;
		IntersectTri(ray, tri[i]);
		if (lastT != ray.t) {
			lastIntersect = i;
		}
	}
	for (int i = 0; i < NS; i++) {
		float lastT = ray.t;
		IntersectSphere(ray, spheres[i]);
		if (lastT != ray.t) {
			lastIntersect = i;
			hitSphere = true;
		}
	}
	Hit hit;
	if (lastIntersect != -1) {
		if (hitSphere) {
			hit.type = SPHERE;
			hit.index = lastIntersect;
			Sphere sphere = spheres[lastIntersect];
			hit.normal = (ray.O + ray.t * ray.D) - sphere.origin;
			hit.material = sphereMaterials[lastIntersect];
		}
		else {
			hit.type = TRIANGLE;
			hit.index = lastIntersect;
			Tri triangle = tri[lastIntersect];
			hit.normal = cross(triangle.vertex1 - triangle.vertex0, triangle.vertex2 - triangle.vertex0);
			hit.material = diffuseMaterials[lastIntersect];
		}
	}
	else {
		hit.type = NOHIT;
	}
	return hit;
}

const float invPI = 1 / PI;


//float3 Sample(Ray& ray, int depth) {
//
//
//	if (depth > 50) {
//		return float3(0.0f);
//	}
//	Hit hit = Trace(ray);
//#if 0
//	// START DEBUG
//	if (hit.type == NOHIT) {
//		return float(0.0f);
//	}
//	else {
//		return hit.material.albedo;
//	}
//	// END DEBUG
//#endif
//	if (hit.type == NOHIT) {
//		cout << "nohit" << endl;
//		return float3(0.0f);
//	}
//	if (hit.material.type == LIGHT) {
//		cout << "emit" << endl;
//		return hit.material.emittance;
//	}
//	float3 newDirection = UniformSampleHemisphere(hit.normal); // Normalized
//	Ray newRay;
//	newRay.O = ray.O + ray.t * ray.D;
//	newRay.D = newDirection;
//	float3 brdf = hit.material.albedo * invPI;
//	float3 partialIrradiance = 2.0f * PI * dot(normalize(hit.normal), newDirection) * brdf;
//	cout << "a" << depth << endl;
//	float3 newSample = Sample(newRay, depth + 1);
//	cout << "b" << depth << endl;
//	//cout << newSample.x << endl;//<< ', '<< newSample.y << ', ' << newSample.z << endl;
//	return float3(
//		partialIrradiance.x * newSample.x,
//		partialIrradiance.y * newSample.y,
//		partialIrradiance.z * newSample.z
//	);
//}

float3 Sample(Ray& ray, int depth) {

	float3 newSample = (float3)(1.0f, 1.0f, 1.0f);
	while (1) {
		if (depth > 50) {
			return float3(0.0f);
		}

		Hit hit = Trace(ray);
#if 0
		// START DEBUG
		if (hit.type == NOHIT) {
			return float(0.0f);
		}
		else {
			if (hit.material.type == LIGHT) {
				//cout << hit.material.emittance.x << endl;
			}
			return hit.material.albedo;
		}
		// END DEBUG
#endif
		if (hit.type == NOHIT) {
			return float3(0.0f);
		}
		if (hit.material.type == LIGHT) {
			return hit.material.emittance * newSample;
		}
		float3 newDirection = UniformSampleHemisphere(hit.normal); // Normalized
		Ray newRay;
		newRay.O = ray.O + ray.t * ray.D;
		newRay.D = newDirection;
		float3 brdf = hit.material.albedo * invPI;
		float3 partialIrradiance = 2.0f * PI * dot(normalize(hit.normal), newDirection) * brdf;
		//return newDirection;
		ray = newRay;
		//float3 newSample = Sample(newRay, depth + 1);
		newSample = float3(
			partialIrradiance.x * newSample.x,
			partialIrradiance.y * newSample.y,
			partialIrradiance.z * newSample.z
		);
		depth++;
		//return float3(
		//	partialIrradiance.x * newSample.x,
		//	partialIrradiance.y * newSample.y,
		//	partialIrradiance.z * newSample.z
		//);

	}



}

void Assignment2MegakernelApp::TickCPU() {
	// draw the scene
	screen->Clear(0);
	// define the corners of the screen in worldspace
	float3 camera(0, 0, -9);
	float3 p0(-1, 1, -8), p1(1, 1, -8), p2(-1, -1, -8);
	Ray ray;
	Timer t;
	for (int y = 0; y < SCRHEIGHT; y++) for (int x = 0; x < SCRWIDTH; x++)
	{
		// calculate the position of a pixel on the screen in worldspace
		float3 pixelPos = p0 + (p1 - p0) * (x / (float)SCRWIDTH) + (p2 - p0) * (y / (float)SCRHEIGHT);
		// define the ray in worldspace

		float3 accumulator = float3(0.0f);
		for (int i = 0; i < SAMPLES_PER_PIXEL; i++) {
			ray.O = camera;
			ray.D = normalize(pixelPos - ray.O);
			// initially the ray has an 'infinite length'
			ray.t = 1e30f;
			accumulator += Sample(ray, 0);
		}
		float3 color = accumulator / (float)SAMPLES_PER_PIXEL;

		uint r = (uint)(min(color.x, 1.0f) * 255.0);
		uint g = (uint)(min(color.y, 1.0f) * 255.0);
		uint b = (uint)(min(color.z, 1.0f) * 255.0);
		//if (ray.t < 1e30f) screen->Plot(x, y, 0xFFFFFF);
		screen->Plot(x, y, r << 16 | g << 8 | b);
	}
	float elapsed = t.elapsed() * 1000;
	printf("tracing time: %.2fms (%5.2fK rays/s)\n", elapsed, sqr(630) / elapsed);
}

void InitCPU() {
	const float WALL_SIZE = 10.0;

	// Triangles are clockwise

	// Back wall
	tri[0].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[0].vertex2 = float3(WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[0].vertex1 = float3(-WALL_SIZE, WALL_SIZE, WALL_SIZE);
	diffuseMaterials[0].type = DIFFUSE;
	diffuseMaterials[0].albedo = float3(0.6f);
	tri[1].vertex0 = float3(WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[1].vertex1 = float3(-WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[1].vertex2 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	diffuseMaterials[1].type = DIFFUSE;
	diffuseMaterials[1].albedo = float3(0.6f);

	// Floor
	tri[2].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[2].vertex2 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[2].vertex1 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	diffuseMaterials[2].type = DIFFUSE;
	diffuseMaterials[2].albedo = float3(1.0f, 0.0f, 0.0f);
	tri[3].vertex0 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[3].vertex1 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[3].vertex2 = float3(WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	diffuseMaterials[3].type = DIFFUSE;
	diffuseMaterials[3].albedo = float3(1.0f, 0.0f, 0.0f);

	// Left wall
	tri[4].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[4].vertex2 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[4].vertex1 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	diffuseMaterials[4].type = DIFFUSE;
	diffuseMaterials[4].albedo = float3(0.0f, 1.0f, 0.0f);
	tri[5].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[5].vertex1 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	tri[5].vertex2 = float3(-WALL_SIZE, WALL_SIZE, WALL_SIZE);
	diffuseMaterials[5].type = DIFFUSE;
	diffuseMaterials[5].albedo = float3(0.0f, 1.0f, 0.0f);

	// Right wall
	tri[6].vertex0 = float3(WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[6].vertex2 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[6].vertex1 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	diffuseMaterials[6].type = DIFFUSE;
	diffuseMaterials[6].albedo = float3(0.0f, 0.0f, 1.0f);
	tri[7].vertex0 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[7].vertex1 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[7].vertex2 = float3(WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	diffuseMaterials[7].type = DIFFUSE;
	diffuseMaterials[7].albedo = float3(0.0f, 0.0f, 1.0f);

	// Ceiling
	tri[8].vertex0 = float3(-WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[8].vertex2 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[8].vertex1 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	diffuseMaterials[8].type = DIFFUSE;
	diffuseMaterials[8].albedo = float3(1.0f, 0.0f, 1.0f);
	tri[9].vertex0 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[9].vertex1 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	tri[9].vertex2 = float3(WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	diffuseMaterials[9].type = DIFFUSE;
	diffuseMaterials[9].albedo = float3(1.0f, 0.0f, 1.0f);

	// Back wall
	tri[10].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[10].vertex1 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[10].vertex2 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	diffuseMaterials[10].type = DIFFUSE;
	diffuseMaterials[10].albedo = float3(0.6f);
	tri[11].vertex0 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[11].vertex2 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	tri[11].vertex1 = float3(WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	diffuseMaterials[11].type = DIFFUSE;
	diffuseMaterials[11].albedo = float3(0.6f);

	// Left ball
	const float S1_R = 2.0f;
	spheres[0].origin = float3(-S1_R * 1.5f, -WALL_SIZE + S1_R * 0.5f, WALL_SIZE * 0.5f);
	spheres[0].radius = S1_R;
	sphereMaterials[0].type = DIFFUSE;
	sphereMaterials[0].albedo = float3(1.0f, 1.0f, 0.0f);

	// Right ball
	const float S2_R = 2.0f;
	spheres[1].origin = float3(S2_R * 1.5f, -WALL_SIZE + S2_R * 0.5f, WALL_SIZE * 0.5f);
	spheres[1].radius = S2_R;
	sphereMaterials[1].type = DIFFUSE;
	sphereMaterials[1].albedo = float3(0.0f, 1.0f, 1.0f);

	// Lamp
	const float L_R = 4.0f;
	spheres[2].origin = float3(0.0f, WALL_SIZE, 0.0f);
	spheres[2].radius = L_R;
	sphereMaterials[2].type = LIGHT;
	sphereMaterials[2].emittance = float3(2.0f);
}
#endif
#ifdef GPGPU
void InitSpheres() {
	const float WALL_SIZE = 10.0f;

	// Left ball
	const float S1_R = 2.0f;
	spheres[0].ox = -S1_R * 1.5f;
	spheres[0].oy = -WALL_SIZE + S1_R * 0.5f;
	spheres[0].oz = WALL_SIZE * 0.5f;
	//spheres[0].ox = 0.0f;
	//spheres[0].oy = 0.0f;
	//spheres[0].oz = -1.0f;
	spheres[0].radius = S1_R;
	sphereMaterials[0].type = DIFFUSE;
	//sphereMaterials[0].albedo = float3(1.0f, 1.0f, 0.0f);
	sphereMaterials[0].albx = 1.0f, sphereMaterials[0].alby = 1.0f, sphereMaterials[0].albz = 0.0f;


	// Right ball
	const float S2_R = 2.0f;
	spheres[1].ox = S2_R * 1.5f;
	spheres[1].oy = -WALL_SIZE + S2_R * 0.5f;
	spheres[1].oz = WALL_SIZE * 0.5f;
	spheres[1].radius = S2_R;
	sphereMaterials[1].type = DIFFUSE;
	sphereMaterials[1].albx = 0.0f, sphereMaterials[1].alby = 1.0f, sphereMaterials[1].albz = 1.0f;

	// Lamp
	const float L_R = 4.0f;
	spheres[2].ox = 0.0f;
	spheres[2].oy = WALL_SIZE;
	spheres[2].oz = 0.0f;
	spheres[2].radius = L_R;
	sphereMaterials[2].type = LIGHT;
	//sphereMaterials[2].emittance = float3(2.0f);
	sphereMaterials[2].emitx = 2.0f, sphereMaterials[2].emity = 2.0f, sphereMaterials[2].emitz = 2.0f;
	sphereMaterials[2].albx = 2.0f, sphereMaterials[2].alby = 2.0f, sphereMaterials[2].albz = 2.0f;

}

void InitTris() {
	const float WALL_SIZE = 10.0;
	// Triangles are clockwise
	//Back wall
	tri[0] = { -WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE };
	diffuseMaterials[0].type = DIFFUSE;
	diffuseMaterials[0].albx = 0.6f, diffuseMaterials[0].alby = 0.6f, diffuseMaterials[0].albz = 0.6f;

	tri[1] = { WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE };
	diffuseMaterials[1].type = DIFFUSE;
	diffuseMaterials[1].albx = 0.6f, diffuseMaterials[1].alby = 0.6f, diffuseMaterials[1].albz = 0.6f;


	//// Floor
	//tri[2].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	//tri[2].vertex2 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	//tri[2].vertex1 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[2] = { -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, -WALL_SIZE };
	diffuseMaterials[2].type = DIFFUSE;
	diffuseMaterials[2].albx = 1.0f, diffuseMaterials[2].alby = 0.0f, diffuseMaterials[2].albz = 0.0f;

	//tri[3].vertex0 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	//tri[3].vertex1 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	//tri[3].vertex2 = float3(WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[3] = { WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE };
	diffuseMaterials[3].type = DIFFUSE;
	diffuseMaterials[3].albx = 1.0f, diffuseMaterials[3].alby = 0.0f, diffuseMaterials[3].albz = 0.0f;


	//// Left wall
	//tri[4].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	//tri[4].vertex2 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	//tri[4].vertex1 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	tri[4] = { -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE };
	diffuseMaterials[4].type = DIFFUSE;
	diffuseMaterials[4].albx = 0.0f, diffuseMaterials[4].alby = 1.0f, diffuseMaterials[4].albz = 0.0f;

	//tri[5].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	//tri[5].vertex1 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	//tri[5].vertex2 = float3(-WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[5] = { -WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE };
	diffuseMaterials[5].type = DIFFUSE;
	diffuseMaterials[5].albx = 0.0f, diffuseMaterials[5].alby = 1.0f, diffuseMaterials[5].albz = 0.0f;

	//// Right wall
	//tri[6].vertex0 = float3(WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	//tri[6].vertex2 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	//tri[6].vertex1 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[6] = { WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, -WALL_SIZE };
	diffuseMaterials[6].type = DIFFUSE;
	diffuseMaterials[6].albx = 0.0f, diffuseMaterials[6].alby = 0.0f, diffuseMaterials[6].albz = 1.0f;

	//tri[7].vertex0 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	//tri[7].vertex1 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	//tri[7].vertex2 = float3(WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	tri[7] = { WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE };
	diffuseMaterials[7].type = DIFFUSE;
	diffuseMaterials[7].albx = 0.0f, diffuseMaterials[7].alby = 0.0f, diffuseMaterials[7].albz = 1.0f;

	//// Ceiling
	//tri[8].vertex0 = float3(-WALL_SIZE, WALL_SIZE, WALL_SIZE);
	//tri[8].vertex2 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	//tri[8].vertex1 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	tri[8] = { -WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE };
	diffuseMaterials[8].type = DIFFUSE;
	diffuseMaterials[8].albx = 1.0f, diffuseMaterials[8].alby = 0.0f, diffuseMaterials[8].albz = 1.0f;

	//tri[9].vertex0 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	//tri[9].vertex1 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	//tri[9].vertex2 = float3(WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	tri[9] = { WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE };
	diffuseMaterials[9].type = DIFFUSE;
	diffuseMaterials[9].albx = 1.0f, diffuseMaterials[9].alby = 0.0f, diffuseMaterials[9].albz = 1.0f;

	//// Back wall
	//tri[10].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	//tri[10].vertex1 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	//tri[10].vertex2 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	tri[10] = { -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE };
	diffuseMaterials[10].type = DIFFUSE;
	diffuseMaterials[10].albx = 0.6f, diffuseMaterials[10].alby = 0.6f, diffuseMaterials[10].albz = 0.6f;

	//tri[11].vertex0 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	//tri[11].vertex2 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	//tri[11].vertex1 = float3(WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	tri[11] = { WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE };
	diffuseMaterials[11].type = DIFFUSE;
	diffuseMaterials[11].albx = 0.6f, diffuseMaterials[11].alby = 0.6f, diffuseMaterials[1].albz = 0.6f;
}

void InitOpenCL() {
#ifdef ANALYZE_RESULTS
	timefile.open(filename.c_str());
	timefile << "frameidx," << "time," << "rays" << "\n";
	timefile.close();
#endif

	InitSpheres();
	InitTris();

	float3 camera(0, 0, -9);
	float3 p0(-1, 1, -8), p1(1, 1, -8), p2(-1, -1, -8);
	if (!kernel)
	{
		Kernel::InitCL();

		kernel = new Kernel("cl/megakernel.cl", "render");
		clbuf_r = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(int), cl_r, Buffer::DEFAULT);
		clbuf_g = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(int), cl_g, Buffer::DEFAULT);
		clbuf_b = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(int), cl_b, Buffer::DEFAULT);

		clbuf_spheres = new Buffer(NS * sizeof(Sphere), spheres, Buffer::DEFAULT);
		clbuf_tris = new Buffer(N * sizeof(Tri), tri, Buffer::DEFAULT);
		clbuf_mat_sphere = new Buffer(NS * sizeof(DiffuseMat), sphereMaterials, Buffer::DEFAULT);
		clbuf_mat_tri = new Buffer(N * sizeof(DiffuseMat), diffuseMaterials, Buffer::DEFAULT);
	}
	kernel->SetArguments(clbuf_r, clbuf_g, clbuf_b, clbuf_spheres, clbuf_tris, clbuf_mat_sphere, clbuf_mat_tri, SCRWIDTH, SCRHEIGHT, SAMPLES_PER_PIXEL, camera, p0, p1, p2);
	clbuf_r->CopyToDevice();
	clbuf_g->CopyToDevice();
	clbuf_b->CopyToDevice();
	clbuf_spheres->CopyToDevice();
	clbuf_tris->CopyToDevice();
	clbuf_mat_sphere->CopyToDevice();
	clbuf_mat_tri->CopyToDevice();

}
#endif


void Assignment2MegakernelApp::Init()
{
#ifdef GPGPU
	InitOpenCL();
#endif
#ifdef CPU
	InitCPU();
#endif
}


void Assignment2MegakernelApp::TickOpenCL() {
	// draw the scene
	screen->Clear(0);

	// define the corners of the screen in worldspace

	//Ray ray;
	Timer t;

	kernel->Run(SCRWIDTH * SCRHEIGHT);
	clbuf_r->CopyFromDevice();
	clbuf_g->CopyFromDevice();
	clbuf_b->CopyFromDevice();


	//cout << cl_r[0] << endl;
	for (int y = 0; y < SCRHEIGHT; y++) for (int x = 0; x < SCRWIDTH; x++) {
		uint idx = y * SCRWIDTH + x;
		//if (cl_r[idx] > 0) cout << "aa" << endl;
		screen->Plot(x, y, cl_r[idx] << 16 | cl_g[idx] << 8 | cl_b[idx]);
	}
	//cout << "rendered 1 frame" << endl;
	float elapsed = t.elapsed() * 1000;
	printf("tracing time: %.2fms (%5.2fK rays/s)\n", elapsed, sqr(630) / elapsed);

#ifdef ANALYZE_RESULTS
	timefile.open(filename.c_str(), std::ios_base::app);
	timefile << frameidx << "," << elapsed <<  "," << sqr(630) / elapsed << "\n";
	timefile.close();
	frameidx++;
#endif

}


void Assignment2MegakernelApp::Tick(float deltaTime)
{
#ifdef GPGPU
	TickOpenCL();
#endif
#ifdef CPU
	TickCPU();
#endif


}