#include "precomp.h"
#include "Assignment2CPU.h"
#include <iostream>

// THIS SOURCE FILE:
// Code for the article "How to Build a BVH", part 1: basics. Link:
// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics
// This is bare-bones BVH construction and traversal code, running in
// a minimalistic framework.
// Feel free to copy this code to your own framework. Absolutely no
// rights are reserved. No responsibility is accepted either.
// For updates, follow me on twitter: @j_bikker.

TheApp* CreateApp() { return new Assignment2CPUApp(); }

// triangle count
//#define N	10
//#define NS  3
#define N 12
#define NS 2
#define SAMPLES_PER_PIXEL 10
//#define USE_NEE

// forward declarations

// minimal structs
struct Sphere { float3 origin; float radius; };
struct Tri { float3 vertex0, vertex1, vertex2; float3 centroid; };

enum MaterialType {
	DIFFUSE,
	LIGHT,
	MIRROR,
};

struct Material {
	MaterialType type;
	union { float3 albedo; float3 emittance; };
};
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
	Material material;
};

// application data
Tri tri[N];
Material triangleMaterials[N];

int triangleLightSize;
uint triangleLights[N];

Sphere spheres[NS];
Material sphereMaterials[NS];

int sphereLightSize;
uint sphereLights[NS];

// functions

void IntersectTri( Ray& ray, const Tri& tri )
{
	const float3 edge1 = tri.vertex1 - tri.vertex0;
	const float3 edge2 = tri.vertex2 - tri.vertex0;
	const float3 h = cross( ray.D, edge2 );
	const float a = dot( edge1, h );
	if (a > -0.0001f && a < 0.0001f) return; // ray parallel to triangle
	const float f = 1 / a;
	const float3 s = ray.O - tri.vertex0;
	const float u = f * dot( s, h );
	if (u < 0 || u > 1) return;
	const float3 q = cross( s, edge1 );
	const float v = f * dot( ray.D, q );
	if (v < 0 || u + v > 1) return;
	const float t = f * dot( edge2, q );
	if (t > 0.0001f) ray.t = min( ray.t, t );
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
		result = float3(RandomFloat()*2.0f - 1.0f, RandomFloat() * 2.0f - 1.0f, RandomFloat() * 2.0f - 1.0f);
	} while (sqrLength(result) > 1);
	if (dot(result, normal) < 0) {
		result = -result;
	}
	// Normalize result
	return normalize(result);
}

struct LightCheck {
	Material light;
	float3 L; // Normalized
	float3 Nl; // Normalized
	float dist;
	float A;
};

LightCheck RandomPointOnLight(float3 src) {
	int totalLights = triangleLightSize + sphereLightSize;
	int selectedLight = (int) (RandomFloat() * totalLights);
	LightCheck lightCheck;
	if (selectedLight < triangleLightSize) {
		Tri selectedTri = tri[triangleLights[selectedLight]];
		Material selectedMat = triangleMaterials[triangleLights[selectedLight]];
		lightCheck.light = selectedMat;
		float u = RandomFloat();
		float v = RandomFloat();
		float uSqrt = sqrtf(u);
		float3 triCross = cross(selectedTri.vertex1 - selectedTri.vertex0, selectedTri.vertex2 - selectedTri.vertex0);
		float3 I = (1 - uSqrt) * selectedTri.vertex0 + uSqrt * (1 - v) * selectedTri.vertex1 + uSqrt * v * selectedTri.vertex2;
		lightCheck.L = I;
		lightCheck.Nl = normalize(triCross);
		lightCheck.dist = length(I - src);
		lightCheck.A = 0.5f * length(triCross);
	}
	else {
		// Selecting sphere
		selectedLight -= triangleLightSize;
		Sphere selectedSphere = spheres[sphereLights[selectedLight]];
		float3 direction = src - selectedSphere.origin;
		float3 I = UniformSampleHemisphere(direction);
		float3 intersectDirection = I - src;
		lightCheck.light = sphereMaterials[sphereLights[selectedLight]];
		lightCheck.L = normalize(intersectDirection);
		lightCheck.Nl = normalize(I - selectedSphere.origin);
		lightCheck.dist = length(intersectDirection);
		lightCheck.A = 2 * PI * selectedSphere.radius * selectedSphere.radius;
	}
	return lightCheck;
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
			hit.material = triangleMaterials[lastIntersect];
		}
	}
	else {
		hit.type = NOHIT;
	}
	return hit;
}

const float invPI = 1 / PI;

float3 Sample(Ray& ray, int depth, bool lastSpecular) {
	if (depth > 10) {
		return float3(0.0f);
	}
	// Trace ray
	Hit hit = Trace(ray);
	float3 normal = normalize(hit.normal);
	float3 I = ray.O + ray.t * ray.D;
#if 0
	// START DEBUG
	if (hit.type == NOHIT) {
		return float(0.0f);
	}
	else {
		return hit.material.albedo;
	}
	// END DEBUG
#endif
	// Terminate if ray left scene
	if (hit.type == NOHIT) {
		return float3(0.0f);
	}
	float3 brdf = hit.material.albedo * invPI;
	// Terminate if we hit a light source
	if (hit.material.type == LIGHT) {
#ifdef USE_NEE
		if (lastSpecular) {
			return hit.material.emittance;
		}
		else {
			return float3(0.0f);
		}
#else
		return hit.material.emittance;
#endif
	}
#ifndef USE_NEE
	if (hit.material.type == MIRROR) {
		// Continue in fixed direction
		Ray newRay;
		newRay.O = I;
		newRay.D = normalize(reflect(ray.D, normal));
		return hit.material.albedo * Sample(newRay, depth + 1, false);
	}
#else
	if (hit.material.type == MIRROR) {
		exit(-1);
	}
#endif
	float3 Ld = 0;
#ifdef USE_NEE
	// Sample a random light source
	LightCheck lightHit = RandomPointOnLight(I);
	Ray shadowRay;
	shadowRay.O = I;
	shadowRay.D = lightHit.L;
	shadowRay.t = lightHit.dist;
	// Cast shadow ray
	Trace(shadowRay);
	if (dot(normal, lightHit.L) > 0 && dot(lightHit.Nl, -lightHit.L) > 0) if (shadowRay.t == lightHit.dist) {
		float solidAngle = (dot(lightHit.Nl, -lightHit.L) * lightHit.A) / (lightHit.dist * lightHit.dist);
		Ld = lightHit.light.emittance * solidAngle * dot(normal, lightHit.L) * (triangleLightSize + sphereLightSize);
		Ld = float3(
			brdf.x * Ld.x,
			brdf.y * Ld.y,
			brdf.z * Ld.z
		);
	}
#endif
	// Continue in random direction
	float3 newDirection = UniformSampleHemisphere(normal); // Normalized
	Ray newRay;
	newRay.O = I;
	newRay.D = newDirection;
	// Update throughput
	float3 Ei = Sample(newRay, depth + 1, false) * dot(normal, newDirection);
	return PI * 2.0f * brdf * Ei + Ld;
}

void Assignment2CPUApp::Init()
{
	const float WALL_SIZE = 10.0;

	// Triangles are clockwise

	// Back wall
	tri[0].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[0].vertex2 = float3(WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[0].vertex1 = float3(-WALL_SIZE, WALL_SIZE, WALL_SIZE);
	triangleMaterials[0].type = DIFFUSE;
	triangleMaterials[0].albedo = float3(1.0f);
	tri[1].vertex0 = float3(WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[1].vertex1 = float3(-WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[1].vertex2 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	triangleMaterials[1].type = DIFFUSE;
	triangleMaterials[1].albedo = float3(1.0f);

	// Floor
	tri[2].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[2].vertex2 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[2].vertex1 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	triangleMaterials[2].type = DIFFUSE;
	triangleMaterials[2].albedo = float3(1.0f);
	tri[3].vertex0 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[3].vertex1 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[3].vertex2 = float3(WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	triangleMaterials[3].type = DIFFUSE;
	triangleMaterials[3].albedo = float3(1.0f);

	// Left wall
	tri[4].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[4].vertex2 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[4].vertex1 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	triangleMaterials[4].type = DIFFUSE;
	triangleMaterials[4].albedo = float3(1.0f, 0.0f, 0.0f);
	tri[5].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[5].vertex1 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	tri[5].vertex2 = float3(-WALL_SIZE, WALL_SIZE, WALL_SIZE);
	triangleMaterials[5].type = DIFFUSE;
	triangleMaterials[5].albedo = float3(1.0f, 0.0f, 0.0f);

	// Right wall
	tri[6].vertex0 = float3(WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[6].vertex2 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[6].vertex1 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	triangleMaterials[6].type = DIFFUSE;
	triangleMaterials[6].albedo = float3(0.0f, 1.0f, 0.0f);
	tri[7].vertex0 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[7].vertex1 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[7].vertex2 = float3(WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	triangleMaterials[7].type = DIFFUSE;
	triangleMaterials[7].albedo = float3(0.0f, 1.0f, 0.0f);

	// Ceiling
	tri[8].vertex0 = float3(-WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[8].vertex2 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[8].vertex1 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	triangleMaterials[8].type = DIFFUSE;
	triangleMaterials[8].albedo = float3(1.0f);
	tri[9].vertex0 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[9].vertex1 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	tri[9].vertex2 = float3(WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	triangleMaterials[9].type = DIFFUSE;
	triangleMaterials[9].albedo = float3(1.0f);

	// Left ball
	const float S1_R = 2.0f;
	spheres[0].origin = float3(-S1_R*1.5f, -WALL_SIZE + S1_R, WALL_SIZE*0.5f);
	spheres[0].radius = S1_R;
	sphereMaterials[0].type = DIFFUSE;
	sphereMaterials[0].albedo = float3(1.0f, 1.0f, 0.0f);

	// Right ball
	const float S2_R = 2.0f;
	spheres[1].origin = float3(S2_R * 1.5f, -WALL_SIZE + S2_R, WALL_SIZE * 0.5f);
	spheres[1].radius = S2_R;
	sphereMaterials[1].type = DIFFUSE;
	sphereMaterials[1].albedo = float3(0.0f, 1.0f, 1.0f);

	// Lamp
	//const float L_R = 4.0f;
	//spheres[2].origin = float3(0.0f, WALL_SIZE, 0.0f);
	//spheres[2].radius = L_R;
	//sphereMaterials[2].type = LIGHT;
	//sphereMaterials[2].emittance = float3(5.0f);
	const float L_S = 4.0f;
	tri[8].vertex0 = float3(-L_S, WALL_SIZE, L_S);
	tri[8].vertex2 = float3(L_S, WALL_SIZE, L_S);
	tri[8].vertex1 = float3(-L_S, WALL_SIZE, -L_S);
	triangleMaterials[8].type = LIGHT;
	triangleMaterials[8].albedo = float3(4.0f);
	tri[9].vertex0 = float3(L_S, WALL_SIZE, L_S);
	tri[9].vertex1 = float3(-L_S, WALL_SIZE, -L_S);
	tri[9].vertex2 = float3(L_S, WALL_SIZE, -L_S);
	triangleMaterials[9].type = LIGHT;
	triangleMaterials[9].albedo = float3(4.0f);

	// Add all lights to buffer
	for (int i = 0; i < N; i++) {
		Material material = triangleMaterials[i];
		if (material.type == LIGHT) {
			triangleLights[triangleLightSize++] = i;
		}
	}
	for (int i = 0; i < NS; i++) {
		Material material = sphereMaterials[i];
		if (material.type == LIGHT) {
			sphereLights[sphereLightSize++] = i;
		}
	}
}

void Assignment2CPUApp::Tick( float deltaTime )
{
	// draw the scene
	screen->Clear( 0 );
	// define the corners of the screen in worldspace
	float3 camera(0, 0, -15);
	float3 p0( -1, 1, -14 ), p1( 1, 1, -14), p2( -1, -1, -14);
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
			accumulator += Sample(ray, 0, true);
		}
		float3 color = accumulator / (float) SAMPLES_PER_PIXEL;
		uint r = (uint)(min(color.x, 1.0f) * 255.0);
		uint g = (uint)(min(color.y, 1.0f) * 255.0);
		uint b = (uint)(min(color.z, 1.0f) * 255.0);
		//if (ray.t < 1e30f) screen->Plot(x, y, 0xFFFFFF);
		screen->Plot(x, y, r << 16 | g << 8 | b);
	}
	float elapsed = t.elapsed() * 1000;
	printf( "tracing time: %.2fms (%5.2fK rays/s)\n", elapsed, sqr( 630 ) / elapsed );
}

// EOF