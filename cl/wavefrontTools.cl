// Define
//hit types
#define HIT_NOHIT 0
#define HIT_TRIANGLE 1
#define HIT_SPHERE 2

//material types
#define MAT_DIFFUSE 0
#define MAT_LIGHT 1

// REGION STRUCTS
struct Ray {
    float Ox, Oy, Oz;
    float Dx, Dy, Dz;
    float t;
    int startThreadId;
};

float3 RayO(struct Ray* ray) {
    return (float3)(ray->Ox, ray->Oy, ray->Oz);
}

float3 RayD(struct Ray* ray) {
    return (float3)(ray->Dx, ray->Dy, ray->Dz);
}

void SetRayO(struct Ray* ray, float3 O) {
    ray->Ox = O.x;
    ray->Oy = O.y;
    ray->Oz = O.z;
}

void SetRayD(struct Ray* ray, float3 D) {
    ray->Dx = D.x;
    ray->Dy = D.y;
    ray->Dz = D.z;
}

struct ShadowRay {
    struct Ray ray;
    float DEx, DEy, DEz;
};

float3 ShadowRayDE(struct ShadowRay* shadowRay) {
    return (float3)(shadowRay->DEx, shadowRay->DEy, shadowRay->DEz);
}

void SetShadowRayDE(struct ShadowRay* shadowRay, float3 DE) {
    shadowRay->DEx = DE.x;
    shadowRay->DEy = DE.y;
    shadowRay->DEz = DE.z;
}

struct Sphere {
    float ox, oy, oz;
    float radius;
};

float3 SphereOrigin(struct Sphere* sphere) {
    return (float3)(sphere->ox, sphere->oy, sphere->oz);
}

struct Tri {
    float v0x, v0y, v0z;
	float v1x, v1y, v1z;
	float v2x, v2y, v2z;
};

float3 TriVertex0(struct Tri* tri) {
    return (float3)(tri->v0x, tri->v0y, tri->v0z);
}

float3 TriVertex1(struct Tri* tri) {
    return (float3)(tri->v1x, tri->v1y, tri->v1z);
}

float3 TriVertex2(struct Tri* tri) {
    return (float3)(tri->v2x, tri->v2y, tri->v2z);
}

struct Material {
	int type;
	float albedoX, albedoY, albedoZ;
};

float3 MaterialAlbedo(struct Material* material) {
    return (float3)(material->albedoX, material->albedoY, material->albedoZ);
}

void SetMaterialAlbedo(struct Material* material, float3 albedo) {
    material->albedoX = albedo.x;
    material->albedoY = albedo.y;
    material->albedoZ = albedo.z;
}

struct Hit {
	int type; //NOHIT = 0, TRIANGLE = 1, SPHERE = 2
	int index;
	float normalX, normalY, normalZ;
	struct Material material;
};

float3 HitNormal(struct Hit* hit) {
    return (float3)(hit->normalX, hit->normalY, hit->normalZ);
}

void SetHitNormal(struct Hit* hit, float3 normal) {
    hit->normalX = normal.x;
    hit->normalY = normal.y;
    hit->normalZ = normal.z;
}
// ENDREGION STRUCTS

// REGION FUNCTIONS

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
    //ray->t = 0.0;
	float t = f * dot( edge2, q );
	if (t > 0.0001f) ray->t = min( ray->t, t );
}


// ENDREGION FUNCTIONS
