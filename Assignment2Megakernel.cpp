#include "precomp.h"
#include "Assignment2Megakernel.h"
#include <iostream>

// Feel free to copy this code to your own framework. Absolutely no
// rights are reserved. No responsibility is accepted either.
// For updates, follow me on twitter: @j_bikker.

TheApp* CreateApp() { return new Assignment2MegakernelApp(); }

#define N	10 // triangle count  <-- adjust in megakernel.cl as well
#define SPHERE_AMT 5 //<<-adjust this value in megakernel.cl as well
#define NS (SPHERE_AMT*SPHERE_AMT+1)
#define SAMPLES_PER_PIXEL 200

static Kernel* kernel = 0; //megakernel
static Buffer* clbuf_spheres = 0; // buffer for spheres
static Buffer* clbuf_tris = 0; // buffer for triangles
static Buffer* clbuf_mat_sphere = 0; // buffer for sphere materials
static Buffer* clbuf_mat_tri = 0; // buffer for sphere materials
static Buffer* clbuf_rand_seed = 0;
static Buffer* clbuf_accumulator = 0;

cl_float4 cl_accumulator[SCRWIDTH * SCRHEIGHT];
uint seeds[SCRWIDTH * SCRHEIGHT];

const float WALL_SIZE = 10.0;
const float3 camera(0, 0, -(2 * WALL_SIZE));
const float3 p0(-WALL_SIZE, WALL_SIZE, -WALL_SIZE), p1(WALL_SIZE, WALL_SIZE, -WALL_SIZE), p2(-WALL_SIZE, -WALL_SIZE, -WALL_SIZE);

// forward declarations

// minimal structs
struct Sphere { float ox, oy, oz; float radius; };
struct Tri {
	float v0x, v0y, v0z;
	float v1x, v1y, v1z;
	float v2x, v2y, v2z;
};

#define DIFFUSE 0
#define LIGHT 1
struct DiffuseMat {
	int type;
	float albx, alby, albz;
	float emitx, emity, emitz;
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
	DiffuseMat material;
};

// application data
Tri tri[N];
DiffuseMat diffuseMaterials[N];

Sphere spheres[NS];
DiffuseMat sphereMaterials[NS];

uint WangHash(uint s)
{
	s = (s ^ 61) ^ (s >> 16);
	s *= 9, s = s ^ (s >> 4);
	s *= 0x27d4eb2d;
	s = s ^ (s >> 15);
	return s;
}

void InitSeeds() {
	for (int i = 0; i < SCRWIDTH * SCRHEIGHT; i++) {
		seeds[i] = WangHash((i + 1) * 17);
	}
}

void InitSpheres() {
	const float S_R = 1.0f;
	const float offset = 1.0f;
	int i = 0;
	for (int z = 0; z < SPHERE_AMT; z++) {
		for (int x = 0; x < SPHERE_AMT; x++, i++) {
			spheres[i].ox = (float)(3 * S_R * (x - SPHERE_AMT / 2));
			spheres[i].oy = -WALL_SIZE + S_R;
			spheres[i].oz = (float)(3 * S_R * (z - SPHERE_AMT / 2));
			spheres[i].radius = S_R;
			sphereMaterials[i].type = DIFFUSE;
			sphereMaterials[i].albx = RandomFloat() * 0.5 + 0.5;
			sphereMaterials[i].alby = RandomFloat() * 0.5 + 0.5;
			sphereMaterials[i].albz = RandomFloat() * 0.5 + 0.5;
			cout << i << endl;
		}
	}
	const float L_R = 4.0f;
	spheres[i].ox = 0.0f;;
	spheres[i].oy = WALL_SIZE;
	spheres[i].oz = 0.0f;
	spheres[i].radius = L_R;
	sphereMaterials[i].type = LIGHT;
	float intensity = 6.0f;
	sphereMaterials[i].emitx = intensity, sphereMaterials[i].emity = intensity, sphereMaterials[i].emitz = intensity;
	sphereMaterials[i].albx = intensity, sphereMaterials[i].alby = intensity, sphereMaterials[i].albz = intensity;
}

void InitTris() {
	// Triangles are clockwise
	//Back wall
	tri[0] = { -WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE };
	diffuseMaterials[0].type = DIFFUSE;
	diffuseMaterials[0].albx = 1.0f, diffuseMaterials[0].alby = 1.0f, diffuseMaterials[0].albz = 1.0f;

	tri[1] = { WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE };
	diffuseMaterials[1].type = DIFFUSE;
	diffuseMaterials[1].albx = 1.0f, diffuseMaterials[1].alby = 1.0f, diffuseMaterials[1].albz = 1.0f;

	//// Floor
	tri[2] = { -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, -WALL_SIZE };
	diffuseMaterials[2].type = DIFFUSE;
	diffuseMaterials[2].albx = 1.0f, diffuseMaterials[2].alby = 1.0f, diffuseMaterials[2].albz = 1.0f;

	tri[3] = { WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE };
	diffuseMaterials[3].type = DIFFUSE;
	diffuseMaterials[3].albx = 1.0f, diffuseMaterials[3].alby = 1.0f, diffuseMaterials[3].albz = 1.0f;

	//// Left wall
	tri[4] = { -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE };
	diffuseMaterials[4].type = DIFFUSE;
	diffuseMaterials[4].albx = 1.0f, diffuseMaterials[4].alby = 0.0f, diffuseMaterials[4].albz = 0.0f;

	tri[5] = { -WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE };
	diffuseMaterials[5].type = DIFFUSE;
	diffuseMaterials[5].albx = 1.0f, diffuseMaterials[5].alby = 0.0f, diffuseMaterials[5].albz = 0.0f;

	//// Right wall
	tri[6] = { WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, -WALL_SIZE };
	diffuseMaterials[6].type = DIFFUSE;
	diffuseMaterials[6].albx = 0.0f, diffuseMaterials[6].alby = 1.0f, diffuseMaterials[6].albz = 0.0f;

	tri[7] = { WALL_SIZE, -WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE };
	diffuseMaterials[7].type = DIFFUSE;
	diffuseMaterials[7].albx = 0.0f, diffuseMaterials[7].alby = 1.0f, diffuseMaterials[7].albz = 0.0f;

	//// Ceiling
	tri[8] = { -WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, WALL_SIZE };
	diffuseMaterials[8].type = DIFFUSE;
	diffuseMaterials[8].albx = 1.0f, diffuseMaterials[8].alby = 1.0f, diffuseMaterials[8].albz = 1.0f;

	tri[9] = { WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, -WALL_SIZE, WALL_SIZE, WALL_SIZE, -WALL_SIZE };
	diffuseMaterials[9].type = DIFFUSE;
	diffuseMaterials[9].albx = 1.0f, diffuseMaterials[9].alby = 1.0f, diffuseMaterials[9].albz = 1.0f;
}

void InitOpenCL(Surface* screen) {
	InitSeeds();
	InitSpheres();
	InitTris();

	if (!kernel)
	{
		Kernel::InitCL();

		kernel = new Kernel("cl/megakernel.cl", "render");

		clbuf_spheres = new Buffer(NS * sizeof(Sphere), spheres, Buffer::DEFAULT);
		clbuf_tris = new Buffer(N * sizeof(Tri), tri, Buffer::DEFAULT);
		clbuf_mat_sphere = new Buffer(NS * sizeof(DiffuseMat), sphereMaterials, Buffer::DEFAULT);
		clbuf_mat_tri = new Buffer(N * sizeof(DiffuseMat), diffuseMaterials, Buffer::DEFAULT);
		clbuf_rand_seed = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(uint), seeds, Buffer::DEFAULT);
		clbuf_accumulator = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(cl_float4), cl_accumulator, Buffer::DEFAULT);
	}
	kernel->SetArguments(clbuf_spheres, clbuf_tris, clbuf_mat_sphere, clbuf_mat_tri, SCRWIDTH, SCRHEIGHT, SAMPLES_PER_PIXEL, camera, p0, p1, p2, clbuf_rand_seed, clbuf_accumulator);
	clbuf_spheres->CopyToDevice();
	clbuf_tris->CopyToDevice();
	clbuf_mat_sphere->CopyToDevice();
	clbuf_mat_tri->CopyToDevice();
	clbuf_rand_seed->CopyToDevice();
}

void Assignment2MegakernelApp::Init()
{
	InitOpenCL(screen);
}

void Assignment2MegakernelApp::TickOpenCL() {
	// draw the scene
	screen->Clear(0);

	Timer t;

	for (int i = 0; i < SAMPLES_PER_PIXEL; i++) {
		kernel->Run(SCRWIDTH * SCRHEIGHT);
	}

	clbuf_accumulator->CopyFromDevice();

	for (int i = 0; i < SCRWIDTH * SCRHEIGHT; i++) {
		float3 color = float3(cl_accumulator[i].x, cl_accumulator[i].y, cl_accumulator[i].z);
		color /= SAMPLES_PER_PIXEL;

		int ri = int(min(1.0f, color.x) * 255.0f);
		int gi = int(min(1.0f, color.y) * 255.0f);
		int bi = int(min(1.0f, color.z) * 255.0f);

		int x = i % SCRWIDTH;
		int y = i / SCRWIDTH;

		screen->Plot(x, y, ri << 16 | gi << 8 | bi);

		cl_accumulator[i] = { 0.0f, 0.0f, 0.0f, 0.0f };
	}

	clbuf_accumulator->CopyToDevice();

	float elapsed = t.elapsed() * 1000;
	printf("tracing time: %.2fms (%5.2fK rays/s)\n", elapsed, sqr(630) / elapsed);
}

void Assignment2MegakernelApp::Tick(float deltaTime)
{
	TickOpenCL();
}