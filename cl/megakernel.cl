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
    float v0x, v0y, v0z;
	float v1x, v1y, v1z;
	float v2x, v2y, v2z;
};

struct DiffuseMat {
	int type;
	float albx, alby, albz;
	float emitx, emity, emitz;
};

struct Hit {
	int type; //NOHIT = 0, TRIANGLE = 1, SPHERE = 2
	int index;
	float3 normal;
	struct DiffuseMat material;
};
////
//GLOBAL VARIABLES
// __constant int N = 12; //number of tris
// __constant int NS = 3; //number of spheres
__constant int N = 10; // triangle count
__constant int SPHERE_AMT = 5;
__constant int NS = (SPHERE_AMT*SPHERE_AMT+1);
////
//FUNCTIONS
//HELPER FUNCTIONS

// using random numbers in GPU code:
// 1. seed using the thread id and a Wang Hash: seed = WangHash( (threadidx+1)*17 )
// 2. from there on: use RandomInt / RandomFloat
uint WangHash( uint s ) 
{ 
	s = (s ^ 61) ^ (s >> 16);
	s *= 9, s = s ^ (s >> 4);
	s *= 0x27d4eb2d;
	s = s ^ (s >> 15); 
	return s; 
}
uint RandomInt( uint* s ) // Marsaglia's XOR32 RNG
{ 
	*s ^= *s << 13;
	*s ^= *s >> 17;
	* s ^= *s << 5; 
	return *s; 
}
float RandomFloat( uint* s ) 
{ 
	return RandomInt( s ) * 2.3283064365387e-10f; // = 1 / (2^32-1)
}

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
	float3 vertex0 = (float3)(tri->v0x, tri->v0y, tri->v0z);
	float3 vertex1 = (float3)(tri->v1x, tri->v1y, tri->v1z);
	float3 vertex2 = (float3)(tri->v2x, tri->v2y, tri->v2z);
	float3 edge1 = vertex1 - vertex0;
	float3 edge2 = vertex2 - vertex0;
	float3 h = cross( ray->D, edge2 );
	float a = dot( edge1, h );
	if (a > -0.0001f && a < 0.0001f) return; // ray parallel to triangle
	float f = 1 / a;
	float3 s = ray->O - vertex0;
	float u = f * dot( s, h );
	if (u < 0 || u > 1) return;
	float3 q = cross( s, edge1 );
	float v = f * dot( ray->D, q );
	if (v < 0 || u + v > 1) return;
    //ray->t = 0.0;
	float t = f * dot( edge2, q );
	if (t > 0.0001f) ray->t = min( ray->t, t );
}

// Returns a normalized vector of which the angle is on the same hemisphere as the normal.
float3 UniformSampleHemisphere(float3 normal, uint* seed) {
	float3 result;
	do {
		result = (float3)(RandomFloat(seed)*2.0f - 1.0f, RandomFloat(seed) * 2.0f - 1.0f, RandomFloat(seed) * 2.0f - 1.0f);
	} while (length(result) > 1);
	if (dot(result, normal) < 0) {
		result = -result;
	}
	// Normalize result
	return normalize(result);
}

struct Hit Trace(struct Ray* ray,__global struct Sphere* spheres, __global struct Tri* tri,
__global struct DiffuseMat* sphereMaterials, __global struct DiffuseMat* triMaterials) {
	int lastIntersect = -1;
	bool hitSphere = false;

	//float lastT = 1e30f;

	for (int i = 0; i < N; i++) {
		float lastT = ray->t;
		IntersectTri(ray, &tri[i]);
		if (lastT != ray->t) {
			lastIntersect = i;
		}
	}
	for (int i = 0; i < NS; i++) {
		float lastT = ray->t;
		IntersectSphere(ray, &spheres[i]);
		if (lastT != ray->t) {
		//if(ray->t < 1e30f){
			lastIntersect = i;
			hitSphere = true;
		}
	}

	struct Hit hit;
	//}
	if (lastIntersect != -1) {
		if (hitSphere) {
			hit.type = SPHERE;
			hit.index = lastIntersect;
			struct Sphere sphere = spheres[lastIntersect];
			hit.material = sphereMaterials[lastIntersect];
			hit.normal = (ray->O + ray->t * ray->D) - (float3)(sphere.ox, sphere.oy, sphere.oz);
		}
		else {
			hit.type = TRIANGLE;
			hit.index = lastIntersect;

			struct Tri triangle = tri[lastIntersect];
			float3 vertex0 = (float3)(triangle.v0x, triangle.v0y, triangle.v0z);
			float3 vertex1 = (float3)(triangle.v1x, triangle.v1y, triangle.v1z);
			float3 vertex2 = (float3)(triangle.v2x, triangle.v2y, triangle.v2z);

			hit.material = triMaterials[lastIntersect];
			hit.normal = cross(vertex1 - vertex0, vertex2 - vertex0);
		}
	}
	else {
		hit.type = NOHIT;
	}
	return hit;
}


float3 Sample(struct Ray* ray, __global struct Sphere* spheres, __global struct Tri* tri,
 __global struct DiffuseMat* sphereMaterials, __global struct DiffuseMat* triMaterials, uint* seed) {
	float3 newSample = (float3)(1.0, 1.0, 1.0);
	int depth = 0;
	while (true) {
		if (depth >= 10) {
			return (float3)(0.0f);
		}

		struct Hit hit = Trace(ray, spheres, tri, sphereMaterials, triMaterials);

		// if (hit.type == NOHIT) {
		// 	return (float3)(0.0f);
		// }
		// // if(hit.type == TRIANGLE){
		// // 	return (float3)(255.0f, 0.0f, 0.0f);
		// // }
		// // if(hit.type == SPHERE){
		// // 	return (float3)(0.0f, 255.0f, 0.0f);
		// // }
		// else {
		// 	//struct DiffuseMat hit_mat = hit.material;
		// 	float x = hit.material.albx;
		// 	float y = hit.material.alby;
		// 	float z = hit.material.albz;
		// 	return (float3)(x, y, z);
		// }

		if (hit.type != NOHIT) {
			// if (hit.type == SPHERE) {
			// 	return (float3)(1.0f, 0.0f, 0.0f);
			// }
			// if (hit.type == TRIANGLE) {
			// 	return (float3)(0.0f, 1.0f, 0.0f);
			// }
			if (hit.material.type == LIGHT) {
				float x = hit.material.emitx;
				float y = hit.material.emity;
				float z = hit.material.emitz;
				float3 mat_emmitance = (float3)(x, y, z);
				return mat_emmitance * newSample;
			}
			float3 normal = normalize(hit.normal);
			//return normal;
			float3 newDirection = UniformSampleHemisphere(normal, seed); // Normalized
			struct Ray newRay;
			newRay.O = ray->O + ray->t * ray->D;
			newRay.D = newDirection;
			newRay.t = 1e30f;
			float3 mat_albedo = (float3)(hit.material.albx, hit.material.alby, hit.material.albz);
			float3 brdf = mat_albedo * M_1_PI_F;
			//return brdf;
			//return mat_albedo;

			float3 partialIrradiance = 2.0f * M_PI_F * dot(normal, newDirection) * brdf;
			//return partialIrradiance;

			//float3 newSample = Sample(newRay, depth + 1);

			ray->O = newRay.O;
			ray->D = newRay.D;
			ray->t = newRay.t;

			newSample = (float3)(
				partialIrradiance[0] * newSample[0],
				partialIrradiance[1] * newSample[1],
				partialIrradiance[2] * newSample[2]
			);

			depth++;
		}
		else{
			return (float3)(0.0f);
		}
		//return (float3)(0.0f);
	}
}


//
void Test(struct Ray* ray, __global struct Sphere* spheres){
	for (int i = 0; i < 3; i++) {
		IntersectSphere(ray, &spheres[i]);
	}
}

__kernel void render(
					__global struct Sphere* spheres, __global struct Tri* tri,
					__global struct DiffuseMat* sphereMaterials, __global struct DiffuseMat* triMaterials,
                    const int image_width, const int image_height, const int SAMPLES_PER_PIXEL, 
                    float3 camPos, float3 p0, float3 p1, float3 p2, 
					__global int* rgb, __global uint *seeds){

    int threadIdx = get_global_id(0);

	if (threadIdx >= image_width * image_height) return;
	int x = threadIdx % image_width;
	int y = threadIdx / image_width;
	uint seed =  seeds[threadIdx];//WangHash( (threadIdx+1)*17 );

    struct Ray ray;

	float3 pixelPos = p0 +
		(p1 - p0) * ((float)x / image_width) +
		(p2 - p0) * ((float)y / image_height);
    float3 accumulator = (float3)(0.0f);

	struct Hit hit;

    for (int i = 0; i < SAMPLES_PER_PIXEL; i++) {
		ray.O = camPos;
		ray.D = normalize(pixelPos - camPos);
		// initially the ray has an 'infinite length'
		ray.t = 1e30f;

		accumulator += Sample(&ray, spheres, tri, sphereMaterials, triMaterials, &seed);
	}

	float3 color = accumulator / (float)SAMPLES_PER_PIXEL;


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
    //int red = 0;
    //if(ray.t < 1e30f) red = 255;
    //if(hit.type == SPHERE) red = 255;
	//struct DiffuseMat mt;
	//mt = triMaterials[8];
    // r[threadIdx] = (int)(min(color[0], 1.0f) * 255.0);
    // g[threadIdx] = (int)(min(color[1], 1.0f) * 255.0);
    // b[threadIdx] = (int)(min(color[2], 1.0f) * 255.0);
	seeds[threadIdx] = seed;
	int ri = (int)(min(1.0f, color.x) * 255.0);
    int gi = (int)(min(1.0f, color.y) * 255.0);
    int bi = (int)(min(1.0f, color.z) * 255.0);
	rgb[threadIdx] = ri << 16 | gi << 8 | bi;
}