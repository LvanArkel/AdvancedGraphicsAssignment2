#include "precomp.h"
#include "Assignment2Wavefront.h"
#include <iostream>

// THIS SOURCE FILE:
// Code for the article "How to Build a BVH", part 1: basics. Link:
// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics
// This is bare-bones BVH construction and traversal code, running in
// a minimalistic framework.
// Feel free to copy this code to your own framework. Absolutely no
// rights are reserved. No responsibility is accepted either.
// For updates, follow me on twitter: @j_bikker.

TheApp* CreateApp() { return new Assignment2WavefrontApp(); }

// triangle count
#define N	10
#define SPHERE_AMT 5
#define NS (SPHERE_AMT*SPHERE_AMT+1)

#define SAMPLES_PER_PIXEL 20
//#define USE_NEE

// forward declarations

// minimal structs
struct Sphere {
	float ox, oy, oz;
	float radius;
};
struct Tri {
	float v0x, v0y, v0z;
	float v1x, v1y, v1z;
	float v2x, v2y, v2z;
};

enum MaterialType {
	DIFFUSE,
	LIGHT,
	MIRROR,
};

struct Material {
	MaterialType type;
	float albedoX, albedoY, albedoZ;
};

// application data
// define the corners of the screen in worldspace
const float WALL_SIZE = 10.0;
const float3 camera(0, 0, -(2 * WALL_SIZE));
const float3 p0(-WALL_SIZE, WALL_SIZE, -WALL_SIZE), p1(WALL_SIZE, WALL_SIZE, -WALL_SIZE), p2(-WALL_SIZE, -WALL_SIZE, -WALL_SIZE);
const int rayBufferSize = SCRWIDTH * SCRHEIGHT;


Tri tri[N];
Material triangleMaterials[N];

int triangleLightSize;
uint triangleLights[N];

Sphere spheres[NS];
Material sphereMaterials[NS];

int sphereLightSize;
uint sphereLights[NS];

uint seeds[rayBufferSize];
uint activeRays = 0;

static Kernel* generateKernel;
static Kernel* extendKernel;
static Kernel* shadeKernel;
static Kernel* finalizeKernel;

static Buffer* clbuf_tris;
static Buffer* clbuf_mat_tris;
static Buffer* clbuf_spheres;
static Buffer* clbuf_mat_spheres;

static Buffer* clbuf_rays;
static Buffer* clbuf_hits;
static Buffer* clbuf_new_rays;

static Buffer* clbuf_accumulator;
static Buffer* clbuf_clr;
static Buffer* clbuf_pixels;

static Buffer* clbuf_rand_seed = 0;

static Buffer* clbuf_active_rays;


uint WangHash(uint s)
{
	s = (s ^ 61) ^ (s >> 16);
	s *= 9, s = s ^ (s >> 4);
	s *= 0x27d4eb2d;
	s = s ^ (s >> 15);
	return s;
}

void InitSeeds() {
	for (int i = 0; i < rayBufferSize; i++) {
		seeds[i] = WangHash((i + 1) * 17);
	}
}

void initWalls() {
	// Triangles are clockwise

	// Back wall
	tri[0] = { -WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE };
	triangleMaterials[0] = { MaterialType::DIFFUSE, 1.0f, 1.0f, 1.0f };
	tri[1] = { WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE };
	triangleMaterials[1] = { MaterialType::DIFFUSE, 1.0f, 1.0f, 1.0f };

	// Floor
	tri[2] = { -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, -WALL_SIZE };
	triangleMaterials[2] = { MaterialType::DIFFUSE, 1.0f, 1.0f, 1.0f };
	tri[3] = { WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE };
	triangleMaterials[3] = { MaterialType::DIFFUSE, 1.0f, 1.0f, 1.0f };

	// Left wall
	tri[4] = { -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE };
	triangleMaterials[4] = { MaterialType::DIFFUSE, 1.0f, 0.0f, 0.0f };
	tri[5] = { -WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE };
	triangleMaterials[5] = { MaterialType::DIFFUSE, 1.0f, 0.0f, 0.0f };

	// Right wall
	tri[6] = { WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, -WALL_SIZE };
	triangleMaterials[6] = { MaterialType::DIFFUSE, 0.0f, 1.0f, 0.0f };
	tri[7] = { WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE };
	triangleMaterials[7] = { MaterialType::DIFFUSE, 0.0f, 1.0f, 0.0f };

	// Ceiling
	tri[8] = { -WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE };
	triangleMaterials[8] = { MaterialType::DIFFUSE, 1.0f, 1.0f, 1.0f };
	tri[9] = { WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE };
	triangleMaterials[9] = { MaterialType::DIFFUSE, 1.0f, 1.0f, 1.0f };
}

void initSpheres() {
	const float S_R = 1.0f;
	const float offset = 1.0f;
	for (int z = 0, i = 0; z < SPHERE_AMT; z++) {
		for (int x = 0; x < SPHERE_AMT; x++, i++) {
			spheres[i].ox = (float)(3 * S_R * (x - SPHERE_AMT / 2));
			spheres[i].oy = -WALL_SIZE + S_R;
			spheres[i].oz = (float)(3 * S_R * (z - SPHERE_AMT / 2));
			spheres[i].radius = S_R;
			sphereMaterials[i].type = MaterialType::DIFFUSE;
			sphereMaterials[i].albedoX = RandomFloat() * 0.5 + 0.5;
			sphereMaterials[i].albedoY = RandomFloat() * 0.5 + 0.5;
			sphereMaterials[i].albedoZ = RandomFloat() * 0.5 + 0.5;


		}
	}

}

void initLights() {
	// Lamp
	const float L_R = 4.0f;
	const float L_I = 4.0f;
	spheres[NS-1] = { 0.0f, WALL_SIZE, 0.0f, L_R };
	sphereMaterials[NS-1] = { MaterialType::LIGHT, L_I, L_I, L_I};
	/*const float L_S = 4.0f;
	tri[10] = { -L_S, WALL_SIZE - 0.05f, L_S, L_S, WALL_SIZE - 0.05f, L_S, -L_S, WALL_SIZE - 0.05f, -L_S };
	triangleMaterials[10] = { LIGHT, 4.0f, 4.0f, 4.0f };
	tri[11] = { L_S, WALL_SIZE - 0.05f, L_S, -L_S, WALL_SIZE - 0.05f, -L_S, L_S, WALL_SIZE - 0.05f, -L_S };
	triangleMaterials[11] = { LIGHT, 4.0f, 4.0f, 4.0f };*/

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

void InitScene() {
	initWalls();
	initSpheres();
	initLights();
}



void InitBuffers(Surface* screen) {
	Kernel::InitCL();

	generateKernel = new Kernel("cl/generateKernel.cl", "generate");
	extendKernel = new Kernel("cl/extendKernel.cl", "extend");
	shadeKernel = new Kernel("cl/shadeKernel.cl", "shade");
	finalizeKernel = new Kernel("cl/finalizeKernel.cl", "finalize");

	const int GPURayByteSize = 32;
	const int GPUTriByteSize = 36;
	const int GPUSphereByteSize = 16;
	const int GPUMaterialByteSize = 16;
	const int GPUHitByteSize = 36;

	clbuf_tris = new Buffer(N * GPUTriByteSize, tri, Buffer::READONLY);
	clbuf_mat_tris = new Buffer(N * GPUMaterialByteSize, triangleMaterials, Buffer::READONLY);
	clbuf_spheres = new Buffer(NS * GPUSphereByteSize, spheres, Buffer::READONLY);
	clbuf_mat_spheres = new Buffer(NS * GPUMaterialByteSize, sphereMaterials, Buffer::READONLY);

	clbuf_rays = new Buffer(rayBufferSize * GPURayByteSize);
	clbuf_hits = new Buffer(rayBufferSize * GPUHitByteSize);
	clbuf_new_rays = new Buffer(rayBufferSize * GPURayByteSize);

	clbuf_active_rays = new Buffer(sizeof(uint), &activeRays, Buffer::DEFAULT);

	clbuf_accumulator = new Buffer(rayBufferSize * 4 * sizeof(float));
	clbuf_clr = new Buffer(SCRWIDTH * SCRHEIGHT * 4 * sizeof(cl_float));
	clbuf_pixels = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(uint), screen->pixels, Buffer::DEFAULT);

	clbuf_rand_seed = new Buffer(rayBufferSize * sizeof(uint), seeds, Buffer::DEFAULT);

	generateKernel->SetArguments(
		SCRWIDTH, SCRHEIGHT,
		camera, p0, p1, p2,
		clbuf_rays,
		clbuf_accumulator,
		clbuf_clr
	);
	//extendKernel->SetArguments(
	//	clbuf_tris, N,
	//	clbuf_spheres, NS,
	//	clbuf_mat_tris, clbuf_mat_spheres,
	//	clbuf_rays,
	//	clbuf_hits
	//);
	//shadeKernel->SetArguments(
	//	clbuf_rays,
	//	clbuf_hits,
	//	clbuf_rand_seed,
	//	clbuf_accumulator
	//);

	extendKernel->SetArguments(
		clbuf_tris, N,
		clbuf_spheres, NS,
		clbuf_mat_tris, clbuf_mat_spheres,
		clbuf_rays,
		clbuf_hits
	);
	shadeKernel->SetArguments(
		clbuf_rays,
		clbuf_hits,
		clbuf_rand_seed,
		clbuf_new_rays,
		clbuf_active_rays,
		clbuf_accumulator
	);
	finalizeKernel->SetArguments(
		clbuf_pixels,
		clbuf_accumulator,
		SCRWIDTH * SCRHEIGHT,
		SAMPLES_PER_PIXEL,
		clbuf_clr
	);

	clbuf_tris->CopyToDevice();
	clbuf_mat_tris->CopyToDevice();
	clbuf_spheres->CopyToDevice();
	clbuf_mat_spheres->CopyToDevice();
	clbuf_rand_seed->CopyToDevice();
}

void Assignment2WavefrontApp::Init()
{
	InitScene();
	InitSeeds();
	InitBuffers(screen);
}

void Assignment2WavefrontApp::Tick(float deltaTime)
{
	// draw the scene
	screen->Clear(0);

	Timer t;
	const int maxDepth = 10;
	for (int j = 0; j < SAMPLES_PER_PIXEL; j++) {
		activeRays = SCRWIDTH * SCRHEIGHT;
		//const int lowerRayBound = activeRays / 100;

		generateKernel->Run(SCRWIDTH * SCRHEIGHT);
		extendKernel->SetArgument(6, clbuf_rays);
		shadeKernel->SetArgument(0, clbuf_rays);
		shadeKernel->SetArgument(3, clbuf_new_rays);
		int i;
		
			for (i = 0; i < maxDepth; i++) {
				int activeThreads = activeRays;
				activeRays = 0;

				extendKernel->Run(activeThreads);

				clbuf_active_rays->CopyToDevice();

				shadeKernel->Run(activeThreads);
				clbuf_active_rays->CopyFromDevice();

				//if (clbuf_active_rays == 0) {
				//	break;
				//}
				//printf("Active rays: %d\n", activeRays);
				if (i % 2 == 0) {
					extendKernel->SetArgument(6, clbuf_new_rays);
					shadeKernel->SetArgument(0, clbuf_new_rays);
					shadeKernel->SetArgument(3, clbuf_rays);
				}
				else {
					extendKernel->SetArgument(6, clbuf_rays);
					shadeKernel->SetArgument(0, clbuf_rays);
					shadeKernel->SetArgument(3, clbuf_new_rays);
				}
			}

			//cout << activeRays << endl;

			//swap(clbuf_rays, clbuf_new_rays);
	}
	//if (i % 2 == 1) {
	//	swap(clbuf_rays, clbuf_new_rays);
	//}
	//cout << activeRays << endl;


	finalizeKernel->Run(SCRWIDTH * SCRHEIGHT);
	clbuf_pixels->CopyFromDevice();


	float elapsed = t.elapsed() * 1000;
	printf("tracing time: %.2fms (%5.2fK rays/s)\n", elapsed, sqr(630) / elapsed);
}

// EOF