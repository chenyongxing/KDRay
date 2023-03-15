#pragma once

struct Ray
{
    float3 origin;
    float3 direction;
    float tmax = FLT_MAX;
};

namespace
{
    __forceinline__ __device__ constexpr float origin() { return 1.0f / 32.0f; }
    __forceinline__ __device__ constexpr float float_scale() { return 1.0f / 65536.0f; }
    __forceinline__ __device__ constexpr float int_scale() { return 256.0f; }
    __forceinline__ __device__ float3 offset_ray(const float3 p, const float3 n)
    {
        int3 of_i = make_int3(int_scale() * n.x, int_scale() * n.y, int_scale() * n.z);

        float3 p_i = make_float3(
            __int_as_float(__float_as_int(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
            __int_as_float(__float_as_int(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
            __int_as_float(__float_as_int(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

        return make_float3(
            fabsf(p.x) < origin() ? p.x + float_scale() * n.x : p_i.x,
            fabsf(p.y) < origin() ? p.y + float_scale() * n.y : p_i.y,
            fabsf(p.z) < origin() ? p.z + float_scale() * n.z : p_i.z);
    }
}

struct Interaction
{
    float3 wo;
    float distance;
    float3 position;
    float3 geomNormal;
    // [T B N] faceforward
    float3 normal;
    float3 tangent;
    float3 binormal;
    kdray::MaterialDeviceState material;

    __forceinline__ __device__ Ray spawn_ray(const float3& wi) const
    {
        float3 N = geomNormal;
        if (dot(wi, geomNormal) < 0.0f)
        {
            N = -geomNormal;
        }

        Ray ray;
        ray.origin = offset_ray(position, N);
        ray.direction = wi;
        ray.tmax = FLT_MAX;
        return ray;
    }

    __forceinline__ __device__ Ray spawn_occlus_ray(const float3& wi, float tmax) const
    {
		float3 N = geomNormal;
		if (dot(wi, geomNormal) < 0.0f)
		{
			N = -geomNormal;
		}

        Ray ray;
        ray.origin = position + N * 0.02f;
        ray.direction = wi;
        ray.tmax = tmax;
        return ray;
    }

    __forceinline__ __device__ float3 local_to_world(const float3& dir) const
    {
        return dir.x * tangent + dir.y * binormal + dir.z * normal;
    }

    __forceinline__ __device__ float3 world_to_local(const float3& dir) const
    {
        return make_float3(dot(dir, tangent), dot(dir, binormal), dot(dir, normal));
    }
};

__forceinline__ __device__ void gen_tangent_basis(const float3& normal, float3* tangent, float3* binormal)
{
	const float sign = normal.z >= 0.0f ? 1.0f : -1.0f;
	const float a = -1.0f / (sign + normal.z);
	const float b = normal.x * normal.y * a;
	*tangent = { 1.0f + sign * a * normal.x * normal.x, sign * b, -sign * normal.x };
	*binormal = { b, sign + a * normal.y * normal.y, -normal.y };
}

__forceinline__ __device__ float3 tangent_to_world(const float3& dir, const float3& normal)
{
    float3 tangent;
	float3 binormal;
    gen_tangent_basis(normal, &tangent, &binormal);

	// transform direction
	return dir.x * tangent + dir.y * binormal + dir.z * normal;
}

// ===================================utils=================================
__forceinline__ __device__ float3 sample_texture(const kdray::TextureDevice& texture, const float2& _uv)
{
	float2 uv = _uv;
	uv *= texture.scale;
    uv += texture.offset;
	return make_float3(tex2D<float4>(texture.texture, uv.x, uv.y));
}

// ===================================ray scene intersect=================================
__forceinline__ __device__ void* unpack_pointer(uint32_t i0, uint32_t i1)
{
	const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
	void* ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}

__forceinline__ __device__ void pack_pointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
	const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

__forceinline__ __device__ bool scene_intersect(const Ray& ray, OptixTraversableHandle traverHandle, Interaction* interaction)
{
	unsigned int isectPtr0, isectPtr1;
	pack_pointer(interaction, isectPtr0, isectPtr1);

	optixTrace(
		traverHandle,
		ray.origin,
		ray.direction,
		0.0f,
		ray.tmax,
		0.0f, // rayTime
		0xFF, // visibility Mask
		OPTIX_RAY_FLAG_NONE,
		0, // SBT offset
		1, // SBT stride
		0, // miss SBT Index
		isectPtr0,
		isectPtr1);

	return interaction->distance < FLT_MAX;
}

__forceinline__ __device__ bool scene_intersect_test(const Ray& ray, OptixTraversableHandle traverHandle)
{
	unsigned int occluded(true);
	optixTrace(
		traverHandle,
		ray.origin,
		ray.direction,
		0.0f,
		ray.tmax,
		0.0f,
		0xFF,
		OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
		0,
		1,
		1,
		occluded);

	return !occluded;
}

// ===================================ray shape intersect=================================
static constexpr float eps = 1e-4f;

__forceinline__ __device__ float intersect_sphere(const float3& ray_origin, const float3& ray_direction,
    const float3& center, float radius)
{
    float3 co = center - ray_origin;
	float b = dot(co, ray_direction);
	float det = b * b - dot(co, co) + radius * radius;
	if (det < 0.0f)
		return FLT_MAX;

	det = sqrt(det);
	float t1 = b - det;
	if (t1 > eps)
		return t1;

	float t2 = b + det;
	if (t2 > eps)
		return t2;

	return FLT_MAX;
}

__forceinline__ __device__ float intersect_rect(const float3& ray_origin, const float3& ray_direction,
    const float3& corner, const float3& _u, const float3& _v)
{
    float3 n = normalize(cross(_u, _v));
    float3 u = _u * (1.0f / dot(_u, _u));
    float3 v = _v * (1.0f / dot(_v, _v));

	float dt = dot(ray_direction, n);
	float t = (dot(n, corner) - dot(n, ray_origin)) / dt;
	if (t > eps)
	{
        float3 p = ray_origin + ray_direction * t;
        float3 vi = p - corner;
		float a1 = dot(u, vi);
		if (a1 >= 0.0f && a1 <= 1.0f)
		{
			float a2 = dot(v, vi);
			if (a2 >= 0.0f && a2 <= 1.0f)
				return t;
		}
	}

    return FLT_MAX;
}

__forceinline__ __device__ float intersect_plane(const float3& ray_origin, const float3& ray_direction,
    const float3& point, const float3& normal)
{
    float denom = dot(normal, ray_direction);
    if (denom > eps)
    {
        float3 d = point - ray_origin;
        float t = dot(d, normal) / denom;
        if (t >= 0.0f)
            return t;
    }

	return FLT_MAX;
}

__forceinline__ __device__ float intersect_disk(const float3& ray_origin, const float3& ray_direction,
    const float3& center, const float3& normal, float radius, bool doubleSided)
{
	float t = intersect_plane(ray_origin, ray_direction, center, normal);
    if (doubleSided && t == FLT_MAX)
    {
        t = intersect_plane(ray_origin, ray_direction, center, -normal);
    }
	if (t < FLT_MAX)
    {
        float3 p = ray_origin + ray_direction * t;
        float3 v = p - center;
		float d2 = dot(v, v);
        if (d2 <= radius * radius)
            return t;
	}

    return FLT_MAX;
}
