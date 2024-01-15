//STRUCTS
struct Ray
{
    float3 O, D;
    float t;
};

struct Sphere 
{ 
    float3 origin; 
    float radius; 
};

struct Tri 
{ 
    float3 vertex0, vertex1, vertex2; 
    float3 centroid; 
};

//GLOBAL VARIABLES


//FUNCTIONS
//HELPER FUNCTIONS


//OTHER FUNCTIONS
void IntersectSphere(struct Ray* ray, struct Sphere* sphere) {
	float3 oc = ray->O - sphere->origin;
	float b = dot(oc, ray->D);
	float c = dot(oc, oc) - sphere->radius * sphere->radius;
	float h = b * b - c;
	if (h > 0.0f) {
		h = sqrt(h);
		ray->t = min(ray->t, -b - h);
	}
}


__kernel void render(__global int* r, __global int* g, __global int* b, const int image_width, const int image_height,
	const int SAMPLES_PER_PIXEL, float3 camPos, float3 p0, float3 p1, float3 p2 ){
    int threadIdx = get_global_id(0);

	if (threadIdx >= image_width * image_height) return;
	int x = threadIdx % image_width;
	int y = threadIdx / image_width;

    struct Ray ray;

	float3 pixelPos = ray.O + p0 +
		(p1 - p0) * ((float)x / image_width) +
		(p2 - p0) * ((float)y / image_height);
    float3 accumulator = (float3)(0.0f);

    struct Sphere s0 = {(float3)(0, 0, -1), 0.5};

    for (int i = 0; i < 1; i++) {
		ray.O = camPos;
		ray.D = normalize(pixelPos - ray.O);
		// initially the ray has an 'infinite length'
		ray.t = 1e30f;

        IntersectSphere(&ray, &s0);
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
