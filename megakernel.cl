//DEFINES
//hit types
#define NOHIT 0
#define TRIANGLE 1
#define SPHERE 2

//material types
#define DIFFUSE 0
#define LIGHT 1

//
//STRUCTS
struct Ray
{
    float3 O, D;
    float t;
};

struct Sphere 
{ 
    float ox, oy, oz; 
    float radius; 
};

struct Tri 
{ 
    float3 vertex0, vertex1, vertex2; 
};

struct DiffuseMat {
	int type; //DIFFUSE = 0, LIGHT = 1
	union { float3 albedo; float3 emittance; };
};

struct Hit {
	int type; //NOHIT = 0, TRIANGLE = 1, SPHERE = 2
	int index;
	float3 normal;
	struct DiffuseMat material;
};
////
//GLOBAL VARIABLES

////
//FUNCTIONS
//HELPER FUNCTIONS
uint RandomUInt()
{
	uint seed = 0x12345678;
	seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	return seed;
}
float RandomFloat() { return RandomUInt() * 2.3283064365387e-10f; }
float Rand( float range ) { return RandomFloat() * range; }

////
//OTHER FUNCTIONS
void IntersectSphere(struct Ray* ray, struct Sphere* sphere) {
    float3 sphere_origin = (float3)(sphere->ox, sphere->oy, sphere->oz);
	float3 oc = ray->O - sphere_origin;
	float b = dot(oc, ray->D);
	float c = dot(oc, oc) - sphere->radius * sphere->radius;
	float h = b * b - c;
	if (h > 0.0f) {
		h = sqrt(h);
		ray->t = min(ray->t, -b - h);
	}
}

void IntersectTri(struct Ray* ray, struct Tri* tri )
{
	float3 edge1 = tri->vertex1 - tri->vertex0;
	float3 edge2 = tri->vertex2 - tri->vertex0;
	float3 h = cross( ray->D, edge2 );
	float a = dot( edge1, h );
	if (a > -0.0001 && a < 0.0001) return; // ray parallel to triangle
	float f = 1 / a;
	float3 s = ray->O - tri->vertex0;
	float u = f * dot( s, h );
	if (u < 0 || u > 1) return;
	float3 q = cross( s, edge1 );
	float v = f * dot( ray->D, q );
	if (v < 0 || u + v > 1) return;
    //ray->t = 0.0;
	float t = f * dot( edge2, q );
	if (t > 0.1) ray->t = 0.0;//ray->t = min( ray->t, t );
}

// // Returns a normalized vector of which the angle is on the same hemisphere as the normal.
// float3 UniformSampleHemisphere(float3 normal) {
// 	float3 result;
// 	do {
// 		result = (float3)(RandomFloat()*2.0f - 0.5f, RandomFloat() * 2.0f - 0.5f, RandomFloat() * 2.0f - 0.5f);
// 	} while (length(result) > 1);
// 	if (dot(result, normal) < 0) {
// 		result = -result;
// 	}
// 	// Normalize result
// 	return normalize(result);
// }

// struct Hit Trace(struct Ray* ray) {
// 	int lastIntersect = -1;
// 	bool hitSphere = false;
// 	for (int i = 0; i < N; i++) {
// 		float lastT = ray.t;
// 		IntersectTri(&ray, tri[i]);
// 		if (lastT != ray.t) {
// 			lastIntersect = i;
// 		}
// 	}
// 	// for (int i = 0; i < NS; i++) {
// 	// 	float lastT = ray.t;
// 	// 	IntersectSphere(ray, spheres[i]);
// 	// 	if (lastT != ray.t) {
// 	// 		lastIntersect = i;
// 	// 		hitSphere = true;
// 	// 	}
// 	// }
// 	struct Hit hit;
// 	// if (lastIntersect != -1) {
// 	// 	if (hitSphere) {
// 	// 		hit.type = SPHERE;
// 	// 		hit.index = lastIntersect;
// 	// 		Sphere sphere = spheres[lastIntersect];
// 	// 		hit.normal = (ray.O + ray.t * ray.D) - sphere.origin;
// 	// 		hit.material = sphereMaterials[lastIntersect];
// 	// 	}
// 	// 	else {
// 	// 		hit.type = TRIANGLE;
// 	// 		hit.index = lastIntersect;
// 	// 		Tri triangle = tri[lastIntersect];
// 	// 		hit.normal = cross(triangle.vertex1 - triangle.vertex0, triangle.vertex2 - triangle.vertex0);
// 	// 		hit.material = diffuseMaterials[lastIntersect];
// 	// 	}
// 	// }
// 	// else {
// 	// 	hit.type = NOHIT;
// 	// }
// 	return hit;
// }

////


__kernel void render(__global int* r, __global int* g, __global int* b, __global struct Sphere* spheres, 
                    const int image_width, const int image_height, const int SAMPLES_PER_PIXEL, 
                    float3 camPos, float3 p0, float3 p1, float3 p2 ){

    int threadIdx = get_global_id(0);

	if (threadIdx >= image_width * image_height) return;
	int x = threadIdx % image_width;
	int y = threadIdx / image_width;

    struct Ray ray;

	float3 pixelPos = ray.O + p0 +
		(p1 - p0) * ((float)x / image_width) +
		(p2 - p0) * ((float)y / image_height);
    float3 accumulator = (float3)(0.0f);

    float WALL_SIZE = 10.0f;
    struct Sphere s0;
    //s0.radius = 2.0f;
    //s0.origin = (float3)(-s0.radius * 1.5f, -WALL_SIZE + s0.radius * 0.5f, WALL_SIZE * 0.5f);
    //struct Tri t0 = {(float3)(-WALL_SIZE, -WALL_SIZE, -WALL_SIZE), (float3)(WALL_SIZE, -WALL_SIZE, -WALL_SIZE), (float3)(-WALL_SIZE, WALL_SIZE, -WALL_SIZE)};
    //for (int i = 0; i < SAMPLES_PER_PIXEL; i++) {
    for (int i = 0; i < 1; i++) {
		ray.O = camPos;
		ray.D = normalize(pixelPos - ray.O);
		// initially the ray has an 'infinite length'
		ray.t = 1e30f;

        for (int j = 0; j < 3; j++) {
            //spheres[j].radius = 0.5f;
            IntersectSphere(&ray, &spheres[j]);
        }
        //IntersectSphere(&ray, &s0);
        //IntersectTri(&ray, &t0);
		//accumulator += Sample(ray, 0);
	}

    // calculate the position of a pixel on the screen in worldspace
	//float3 pixelPos = p0 + (p1 - p0) * (x / (float)image_width) + (p2 - p0) * (y / (float)image_height);
	// define the ray in worldspace

	// float3 accumulator = float3(0.0f);
	// for (int i = 0; i < SAMPLES_PER_PIXEL; i++) {
	// 	ray.O = camera;
	// 	ray.D = normalize(pixelPos - ray.O);
	// 	// initially the ray has an 'infinite length'
	// 	ray.t = 1e30f;
	// 	accumulator += Sample(ray, 0);
	// }
	// float3 color = accumulator / (float)SAMPLES_PER_PIXEL;
	// uint r = (uint)(min(color.x, 1.0f) * 255.0);
	// uint g = (uint)(min(color.y, 1.0f) * 255.0);
	// uint b = (uint)(min(color.z, 1.0f) * 255.0);
    int red = 0;
    if(ray.t < 1e30f) red = 255;
    r[threadIdx] = red;
    g[threadIdx] = 0;//0.0f;
    b[threadIdx] = 0;//0.0f;
}
