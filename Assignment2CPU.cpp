#include "precomp.h"
#include "Assignment2CPU.h"

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
#define N	12
#define NS  2

// forward declarations

// minimal structs
struct Sphere { float3 origin; float radius; };
struct Tri { float3 vertex0, vertex1, vertex2; float3 centroid; };
struct DiffuseMat { float3 albedo; };
__declspec(align(32)) struct BVHNode
{
	float3 aabbMin, aabbMax;
	uint leftFirst, triCount;
	bool isLeaf() { return triCount > 0; }
};
struct Ray { float3 O, D; float t = 1e30f; };

// application data
Tri tri[N];
DiffuseMat diffuseMaterials[N];

Sphere spheres[NS];
DiffuseMat sphereMaterials[NS];

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

float3 Color(Ray& ray) {
	int lastIntersect = -1;
	bool checkSpheres = false;
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
			checkSpheres = true;
		}
	}
	if (ray.t < 1e30f) {
		if (checkSpheres) {
			return sphereMaterials[lastIntersect].albedo;
		} else {
			return diffuseMaterials[lastIntersect].albedo;
		}
	}
	else {
		return float3(0.0, 0.0, 0.0);
	}
}

void Assignment2CPUApp::Init()
{
	const float WALL_SIZE = 10.0;

	// Back wall
	tri[0].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[0].vertex1 = float3(WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[0].vertex2 = float3(-WALL_SIZE, WALL_SIZE, WALL_SIZE);
	diffuseMaterials[0].albedo = float3(1.0f);
	tri[1].vertex0 = float3(WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[1].vertex1 = float3(-WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[1].vertex2 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	diffuseMaterials[1].albedo = float3(1.0f);

	// Floor
	tri[2].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[2].vertex1 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[2].vertex2 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	diffuseMaterials[2].albedo = float3(1.0f, 0.0f, 0.0f);
	tri[3].vertex0 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[3].vertex1 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[3].vertex2 = float3(WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	diffuseMaterials[3].albedo = float3(1.0f, 0.0f, 0.0f);

	// Left wall
	tri[4].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[4].vertex1 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[4].vertex2 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	diffuseMaterials[4].albedo = float3(0.0f, 1.0f, 0.0f);
	tri[5].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[5].vertex1 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	tri[5].vertex2 = float3(-WALL_SIZE, WALL_SIZE, WALL_SIZE);
	diffuseMaterials[5].albedo = float3(0.0f, 1.0f, 0.0f);

	// Right wall
	tri[6].vertex0 = float3(WALL_SIZE, -WALL_SIZE, WALL_SIZE);
	tri[6].vertex1 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[6].vertex2 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	diffuseMaterials[6].albedo = float3(0.0f, 0.0f, 1.0f);
	tri[7].vertex0 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[7].vertex1 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[7].vertex2 = float3(WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	diffuseMaterials[7].albedo = float3(0.0f, 0.0f, 1.0f);

	// Ceiling
	tri[8].vertex0 = float3(-WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[8].vertex1 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[8].vertex2 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	diffuseMaterials[8].albedo = float3(1.0f, 0.0f, 1.0f);
	tri[9].vertex0 = float3(WALL_SIZE, WALL_SIZE, WALL_SIZE);
	tri[9].vertex1 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	tri[9].vertex2 = float3(WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	diffuseMaterials[9].albedo = float3(1.0f, 0.0f, 1.0f);

	// Back wall
	tri[10].vertex0 = float3(-WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[10].vertex1 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[10].vertex2 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	diffuseMaterials[10].albedo = float3(1.0f);
	tri[11].vertex0 = float3(WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
	tri[11].vertex1 = float3(-WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	tri[11].vertex2 = float3(WALL_SIZE, WALL_SIZE, -WALL_SIZE);
	diffuseMaterials[11].albedo = float3(1.0f);

	// Left ball
	const float S1_R = 2.0f;
	spheres[0].origin = float3(-S1_R*1.5f, -WALL_SIZE + S1_R*0.5f, WALL_SIZE*0.5f);
	spheres[0].radius = S1_R;
	sphereMaterials[0].albedo = float3(1.0f, 1.0f, 0.0f);

	const float S2_R = 2.0f;
	spheres[1].origin = float3(S2_R * 1.5f, -WALL_SIZE + S2_R * 0.5f, WALL_SIZE * 0.5f);
	spheres[1].radius = S2_R;
	sphereMaterials[1].albedo = float3(0.0f, 1.0f, 1.0f);


	// Right ball

}

void Assignment2CPUApp::Tick( float deltaTime )
{
	// draw the scene
	screen->Clear( 0 );
	// define the corners of the screen in worldspace
	float3 camera(0, 0, -9);
	float3 p0( -1, 1, -8 ), p1( 1, 1, -8), p2( -1, -1, -8);
	Ray ray;
	Timer t;
	for (int y = 0; y < SCRHEIGHT; y++) for (int x = 0; x < SCRWIDTH; x++)
	{
		// calculate the position of a pixel on the screen in worldspace
		float3 pixelPos = p0 + (p1 - p0) * (x / (float)SCRWIDTH) + (p2 - p0) * (y / (float)SCRHEIGHT);
		// define the ray in worldspace
		ray.O = camera;
		ray.D = normalize( pixelPos - ray.O );
		// initially the ray has an 'infinite length'
		ray.t = 1e30f;
		float3 color = Color(ray);
		uint r = (uint)(color.x * 255.0);
		uint g = (uint)(color.y * 255.0);
		uint b = (uint)(color.z * 255.0);
		//if (ray.t < 1e30f) screen->Plot(x, y, 0xFFFFFF);
		screen->Plot(x, y, r << 16 | g << 8 | b);
	}
	float elapsed = t.elapsed() * 1000;
	printf( "tracing time: %.2fms (%5.2fK rays/s)\n", elapsed, sqr( 630 ) / elapsed );
}

// EOF