// Define
//hit types
#define HIT_NOHIT 0
#define HIT_TRIANGLE 1
#define HIT_SPHERE 2

//material types
#define MAT_DIFFUSE 0
#define MAT_LIGHT 1

// STRUCTS
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
