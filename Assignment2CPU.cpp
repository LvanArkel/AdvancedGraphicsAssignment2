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
#define N	64
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
	// intialize a scene with N random triangles
	for (int i = 0; i < N; i++)
	{
		float3 r0 = float3( RandomFloat(), RandomFloat(), RandomFloat() );
		float3 r1 = float3( RandomFloat(), RandomFloat(), RandomFloat() );
		float3 r2 = float3( RandomFloat(), RandomFloat(), RandomFloat() );
		tri[i].vertex0 = r0 * 9 - float3( 5 );
		tri[i].vertex1 = tri[i].vertex0 + r1, tri[i].vertex2 = tri[i].vertex0 + r2;
		diffuseMaterials[i].albedo = float3(RandomFloat(), RandomFloat(), RandomFloat());

	}

	spheres[0] = Sphere{ float3(0.0), 1 };
	sphereMaterials[0] = DiffuseMat{float3(1.0)};
	spheres[1] = Sphere{ float3(2.0, 0.0, 0.0), 0.5 };
	sphereMaterials[1] = DiffuseMat{ float3(0.5) };
}

void Assignment2CPUApp::Tick( float deltaTime )
{
	// draw the scene
	screen->Clear( 0 );
	// define the corners of the screen in worldspace
	float3 camera(0, 0, -18);
	float3 p0( -1, 1, -15 ), p1( 1, 1, -15 ), p2( -1, -1, -15 );
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