#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <cmath>
#include <cstdint>
#include <cfloat>
#include <vector_functions.h>

#define M_PIf		3.14159265358979323846f
#define M_PI_2f		1.57079632679489661923f
#define M_1_PIf		0.318309886183790671538f
#define M_2PIf      6.28318530717958647692f

#if defined(__CUDACC__) || defined(__CUDABE__)
#define KDRAY_INLINE __forceinline__ __device__
#else
#define KDRAY_INLINE inline
#endif

KDRAY_INLINE float sqr(float a)
{
	return a * a;
}

KDRAY_INLINE float rcp(float a)
{
	return 1.0f / a;
}

KDRAY_INLINE float3 sqr(const float3& v)
{
	return make_float3(v.x * v.x, v.y * v.y, v.z * v.z);
}

KDRAY_INLINE float3 sqrt(const float3& v)
{
	return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}

KDRAY_INLINE float3 rcp(const float3& v)
{
	return make_float3(1.0f / v.x, 1.0f / v.y, 1.0f / v.z);
}

KDRAY_INLINE bool isnan(const float3& v)
{
	return isnan(v.x) || isnan(v.y) || isnan(v.z);
}

KDRAY_INLINE bool all_zero(const float3& c)
{
	if (c.x != 0.0f)
		return false;
	if (c.y != 0.0f)
		return false;
	if (c.z != 0.0f)
		return false;
	return true;
}

KDRAY_INLINE float radian(float a)
{
	return a * (M_PIf / 180.0f);
}

KDRAY_INLINE float degree(float a)
{
	return a * (180.0f / M_PIf);
}

KDRAY_INLINE float clamp(const float f, const float a, const float b)
{
	return fmaxf(a, fminf(f, b));
}

KDRAY_INLINE float3 min(const float3& a, const float3& b)
{
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

KDRAY_INLINE float3 max(const float3& a, const float3& b)
{
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

KDRAY_INLINE float2 operator+(const float2& a, const float2& b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}

KDRAY_INLINE float2 operator-(const float2& a, const float2& b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}

KDRAY_INLINE float2 operator/(const float2& a, const float2& b)
{
	return make_float2(a.x / b.x, a.y / b.y);
}

KDRAY_INLINE float2 operator*(const float2& a, const float s)
{
	return make_float2(a.x * s, a.y * s);
}

KDRAY_INLINE float2 operator*(const float s, const float2& a)
{
	return make_float2(a.x * s, a.y * s);
}

KDRAY_INLINE void operator+=(float2& a, const float2& b)
{
	a.x += b.x;
	a.y += b.y;
}

KDRAY_INLINE void operator*=(float2& a, const float2& b)
{
	a.x *= b.x;
	a.y *= b.y;
}

KDRAY_INLINE void operator*=(float2& a, const float s)
{
	a.x *= s;
	a.y *= s;
}

KDRAY_INLINE float3 make_float3(const float s)
{
	return make_float3(s, s, s);
}

KDRAY_INLINE float3 make_float3(const float4& v)
{
	return make_float3(v.x, v.y, v.z);
}

KDRAY_INLINE float3 make_float3(const float2& v0, const float v1) 
{
	return make_float3(v0.x, v0.y, v1);
}

KDRAY_INLINE float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

KDRAY_INLINE float3 operator+(const float3& a, const float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}

KDRAY_INLINE float3 operator+(const float a, const float3& b)
{
	return make_float3(a + b.x, a + b.y, a + b.z);
}

KDRAY_INLINE void operator+=(float3& a, const float3& b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

KDRAY_INLINE float3 operator-(const float3& a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

KDRAY_INLINE float3 operator-(const float3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

KDRAY_INLINE float3 operator-(const float3& a, const float b)
{
	return make_float3(a.x - b, a.y - b, a.z - b);
}

KDRAY_INLINE float3 operator-(const float a, const float3& b)
{
	return make_float3(a - b.x, a - b.y, a - b.z);
}

KDRAY_INLINE float3 operator*(const float3& a, const float s)
{
	return make_float3(a.x * s, a.y * s, a.z * s);
}

KDRAY_INLINE float3 operator*(const float s, const float3& a)
{
	return make_float3(a.x * s, a.y * s, a.z * s);
}

KDRAY_INLINE float3 operator*(const float3& a, const float3& b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

KDRAY_INLINE void operator*=(float3& a, const float3& s)
{
	a.x *= s.x;
	a.y *= s.y;
	a.z *= s.z;
}

KDRAY_INLINE void operator*=(float3& a, const float s)
{
	a.x *= s;
	a.y *= s;
	a.z *= s;
}

KDRAY_INLINE float3 operator/(const float3& a, const float3& b)
{
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

KDRAY_INLINE float3 operator/(const float3& a, const float s)
{
	float inv = 1.0f / s;
	return a * inv;
}

KDRAY_INLINE void operator/=(float3& a, const float s)
{
	a.x /= s;
	a.y /= s;
	a.z /= s;
}

KDRAY_INLINE float3 pow(const float3& a, float b)
{
	return make_float3(powf(a.x, b), powf(a.y, b), powf(a.z, b));
}

KDRAY_INLINE float dot(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

KDRAY_INLINE float3 cross(const float3& a, const float3& b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

KDRAY_INLINE float length(const float3& v)
{
	return sqrtf(dot(v, v));
}

KDRAY_INLINE float3 normalize(const float3& v)
{
	float inv_len = 1.0f / sqrtf(dot(v, v));
	return v * inv_len;
}

KDRAY_INLINE float3 clamp(const float3& v, const float a, const float b)
{
	return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

KDRAY_INLINE float3 clamp(const float3& v, const float3& a, const float3& b)
{
	return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

KDRAY_INLINE float lerp(const float a, const float b, const float t)
{
	return a + t * (b - a);
}

KDRAY_INLINE float3 lerp(const float3& a, const float3& b, const float t)
{
	return a + t * (b - a);
}

KDRAY_INLINE float3 faceforward(const float3& normal, const float3& dir)
{
	float sign = dot(dir, normal) >= 0.0f ? 1.0f : -1.0f;
	return sign * normal;
}

KDRAY_INLINE float3 reflect(const float3& i, const float3& n)
{
	return -i + 2.0f * dot(n, i) * n;
}

KDRAY_INLINE bool refract(const float3& i, const float3& n, float eta, float3* wt)
{
	float cos_i = dot(i, n);
	float sin_i_sq = 1.0f - cos_i * cos_i;
	float sin_t_sq = eta * eta * sin_i_sq;
	// total internal reflection
	if (sin_t_sq >= 1.0f) return false;

	float cos_t = sqrt(1.0f - sin_t_sq);
	*wt = eta * -i + (eta * cos_i - cos_t) * n;
	return true;
}

KDRAY_INLINE float luminance(const float3& rgb)
{
	return dot(rgb, make_float3(0.2126729f, 0.7151522f, 0.0721750f));
}

KDRAY_INLINE float4 make_float4(const float3& v0, const float v1)
{
	return make_float4(v0.x, v0.y, v0.z, v1);
}

KDRAY_INLINE float4 operator+(const float4& a, const float4& b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

KDRAY_INLINE float4 operator-(const float4& a, const float4& b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

KDRAY_INLINE float4 operator*(const float4& a, const float4& s)
{
	return make_float4(a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w);
}

KDRAY_INLINE float4 operator*(const float4& a, const float s)
{
	return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

// right-handed matrix4x4 declarations
struct float4x4
{
	float m[4][4];

	KDRAY_INLINE float3 origin() const
	{
		return make_float3(m[3][0], m[3][1], m[3][2]);
	}

	KDRAY_INLINE float3 basisx() const
	{
		return make_float3(m[0][0], m[0][1], m[0][2]);
	}

	KDRAY_INLINE float3 basisy() const
	{
		return make_float3(m[1][0], m[1][1], m[1][2]);
	}

	KDRAY_INLINE float3 basisz() const
	{
		return make_float3(m[2][0], m[2][1], m[2][2]);
	}

	KDRAY_INLINE float4 operator[](int n) const
	{
		return make_float4(m[n][0], m[n][1], m[n][2], m[n][3]);
	}
};

KDRAY_INLINE float4x4 make_float4x4()
{
	float4x4 m;
	m.m[0][0] = m.m[1][1] = m.m[2][2] = m.m[3][3] = 1.0f;
	m.m[0][1] = m.m[0][2] = m.m[0][3] = 0.0f;
	m.m[1][0] = m.m[1][2] = m.m[1][3] = 0.0f;
	m.m[2][0] = m.m[2][1] = m.m[2][3] = 0.0f;
	m.m[3][0] = m.m[3][1] = m.m[3][2] = 0.0f;
	return m;
}

KDRAY_INLINE float4x4 make_float4x4(
	float t00, float t01, float t02, float t03,
	float t10, float t11, float t12, float t13,
	float t20, float t21, float t22, float t23,
	float t30, float t31, float t32, float t33)
{
	float4x4 m;
	m.m[0][0] = t00; m.m[0][1] = t01; m.m[0][2] = t02; m.m[0][3] = t03;
	m.m[1][0] = t10; m.m[1][1] = t11; m.m[1][2] = t12; m.m[1][3] = t13;
	m.m[2][0] = t20; m.m[2][1] = t21; m.m[2][2] = t22; m.m[2][3] = t23;
	m.m[3][0] = t30; m.m[3][1] = t31; m.m[3][2] = t32; m.m[3][3] = t33;
	return m;
}

KDRAY_INLINE void identity_float4x4(float4x4& m)
{
	m.m[0][0] = m.m[1][1] = m.m[2][2] = m.m[3][3] = 1.0f;
	m.m[0][1] = m.m[0][2] = m.m[0][3] = 0.0f;
	m.m[1][0] = m.m[1][2] = m.m[1][3] = 0.0f;
	m.m[2][0] = m.m[2][1] = m.m[2][3] = 0.0f;
	m.m[3][0] = m.m[3][1] = m.m[3][2] = 0.0f;
}

inline float4x4 translate(const float4x4& m, float x, float y, float z)
{
	float4x4 _m = m;
	_m.m[3][0] = m.m[0][0] * x + m.m[1][0] * y + m.m[2][0] * z + m.m[3][0];
	_m.m[3][1] = m.m[0][1] * x + m.m[1][1] * y + m.m[2][1] * z + m.m[3][1];
	_m.m[3][2] = m.m[0][2] * x + m.m[1][2] * y + m.m[2][2] * z + m.m[3][2];
	return _m;
}

inline float4x4 translate(const float4x4& m, const float3& v)
{
	return translate(m, v.x, v.y, v.z);
}

KDRAY_INLINE float4x4 scale(const float4x4& m, float x, float y, float z)
{
	float4x4 _m = m;
	_m.m[0][0] = m.m[0][0] * x;
	_m.m[0][1] = m.m[0][0] * x;
	_m.m[0][2] = m.m[0][0] * x;
	_m.m[0][3] = m.m[0][0] * x;

	_m.m[1][0] = m.m[0][0] * y;
	_m.m[1][1] = m.m[0][0] * y;
	_m.m[1][2] = m.m[0][0] * y;
	_m.m[1][3] = m.m[0][0] * y;

	_m.m[2][0] = m.m[0][0] * z;
	_m.m[2][1] = m.m[0][0] * z;
	_m.m[2][2] = m.m[0][0] * z;
	_m.m[2][3] = m.m[0][0] * z;
	return _m;
}

KDRAY_INLINE float4x4 scale(const float4x4& m, const float3& v)
{
	return scale(m, v.x, v.y, v.z);
}

KDRAY_INLINE float4x4 lookat(const float3& eye, const float3& center, const float3& up)
{
	float3 f(normalize(center - eye));
	float3 s(normalize(cross(f, up)));
	float3 u(cross(s, f));

	float4x4 m;
	identity_float4x4(m);

	m.m[0][0] = s.x;
	m.m[1][0] = s.y;
	m.m[2][0] = s.z;

	m.m[0][1] = u.x;
	m.m[1][1] = u.y;
	m.m[2][1] = u.z;

	m.m[0][2] = -f.x;
	m.m[1][2] = -f.y;
	m.m[2][2] = -f.z;

	m.m[3][0] = -dot(s, eye);
	m.m[3][1] = -dot(u, eye);
	m.m[3][2] = dot(f, eye);
	return m;
}

KDRAY_INLINE float4x4 perspective(float fovy, float aspect, float zNear, float zFar)
{
	float tanHalfFovy = tanf(fovy / 2.0f);

	float4x4 m;
	identity_float4x4(m);

	m.m[0][0] = 1.0f / (aspect * tanHalfFovy);
	m.m[1][1] = 1.0f / (tanHalfFovy);
	m.m[2][2] = -(zFar + zNear) / (zFar - zNear);
	m.m[2][3] = -1.0f;
	m.m[3][2] = -(2.0f * zFar * zNear) / (zFar - zNear);
	m.m[3][3] = 0.0f;
	return m;
}

KDRAY_INLINE float4x4 operator*(const float4x4& m, float s)
{
	float4x4 _m = m;
	_m.m[0][0] *= s;
	_m.m[0][1] *= s;
	_m.m[0][2] *= s;
	_m.m[0][3] *= s;

	_m.m[1][0] *= s;
	_m.m[1][1] *= s;
	_m.m[1][2] *= s;
	_m.m[1][3] *= s;

	_m.m[2][0] *= s;
	_m.m[2][1] *= s;
	_m.m[2][2] *= s;
	_m.m[2][3] *= s;

	_m.m[3][0] *= s;
	_m.m[3][1] *= s;
	_m.m[3][2] *= s;
	_m.m[3][3] *= s;
	return _m;
}

KDRAY_INLINE float4 operator*(const float4x4& m, const float4& v)
{
	float4 _v;
	_v.x = m.m[0][0] * v.x + m.m[1][0] * v.y + m.m[2][0] * v.z + m.m[3][0] * v.w;
	_v.y = m.m[0][1] * v.x + m.m[1][1] * v.y + m.m[2][1] * v.z + m.m[3][1] * v.w;
	_v.z = m.m[0][2] * v.x + m.m[1][2] * v.y + m.m[2][2] * v.z + m.m[3][2] * v.w;
	_v.w = m.m[0][3] * v.x + m.m[1][3] * v.y + m.m[2][3] * v.z + m.m[3][3] * v.w;

	return _v;
}

KDRAY_INLINE float4x4 operator*(const float4x4& m1, const float4x4& m2)
{
	float4x4 m;
	m.m[0][0] = m1.m[0][0] * m2.m[0][0] + m1.m[1][0] * m2.m[0][1] + m1.m[2][0] * m2.m[0][2] + m1.m[3][0] * m2.m[0][3];
	m.m[0][1] = m1.m[0][1] * m2.m[0][0] + m1.m[1][1] * m2.m[0][1] + m1.m[2][1] * m2.m[0][2] + m1.m[3][1] * m2.m[0][3];
	m.m[0][2] = m1.m[0][2] * m2.m[0][0] + m1.m[1][2] * m2.m[0][1] + m1.m[2][2] * m2.m[0][2] + m1.m[3][2] * m2.m[0][3];
	m.m[0][3] = m1.m[0][3] * m2.m[0][0] + m1.m[1][3] * m2.m[0][1] + m1.m[2][3] * m2.m[0][2] + m1.m[3][3] * m2.m[0][3];

	m.m[1][0] = m1.m[0][0] * m2.m[1][0] + m1.m[1][0] * m2.m[1][1] + m1.m[2][0] * m2.m[1][2] + m1.m[3][0] * m2.m[1][3];
	m.m[1][1] = m1.m[0][1] * m2.m[1][0] + m1.m[1][1] * m2.m[1][1] + m1.m[2][1] * m2.m[1][2] + m1.m[3][1] * m2.m[1][3];
	m.m[1][2] = m1.m[0][2] * m2.m[1][0] + m1.m[1][2] * m2.m[1][1] + m1.m[2][2] * m2.m[1][2] + m1.m[3][2] * m2.m[1][3];
	m.m[1][3] = m1.m[0][3] * m2.m[1][0] + m1.m[1][3] * m2.m[1][1] + m1.m[2][3] * m2.m[1][2] + m1.m[3][3] * m2.m[1][3];

	m.m[2][0] = m1.m[0][0] * m2.m[2][0] + m1.m[1][0] * m2.m[2][1] + m1.m[2][0] * m2.m[2][2] + m1.m[3][0] * m2.m[2][3];
	m.m[2][1] = m1.m[0][1] * m2.m[2][0] + m1.m[1][1] * m2.m[2][1] + m1.m[2][1] * m2.m[2][2] + m1.m[3][1] * m2.m[2][3];
	m.m[2][2] = m1.m[0][2] * m2.m[2][0] + m1.m[1][2] * m2.m[2][1] + m1.m[2][2] * m2.m[2][2] + m1.m[3][2] * m2.m[2][3];
	m.m[2][3] = m1.m[0][3] * m2.m[2][0] + m1.m[1][3] * m2.m[2][1] + m1.m[2][3] * m2.m[2][2] + m1.m[3][3] * m2.m[2][3];

	m.m[3][0] = m1.m[0][0] * m2.m[3][0] + m1.m[1][0] * m2.m[3][1] + m1.m[2][0] * m2.m[3][2] + m1.m[3][0] * m2.m[3][3];
	m.m[3][1] = m1.m[0][1] * m2.m[3][0] + m1.m[1][1] * m2.m[3][1] + m1.m[2][1] * m2.m[3][2] + m1.m[3][1] * m2.m[3][3];
	m.m[3][2] = m1.m[0][2] * m2.m[3][0] + m1.m[1][2] * m2.m[3][1] + m1.m[2][2] * m2.m[3][2] + m1.m[3][2] * m2.m[3][3];
	m.m[3][3] = m1.m[0][3] * m2.m[3][0] + m1.m[1][3] * m2.m[3][1] + m1.m[2][3] * m2.m[3][2] + m1.m[3][3] * m2.m[3][3];
	return m;
}

KDRAY_INLINE float3 transform_direction(const float4x4& m, const float3& v)
{
	float x = m.m[0][0] * v.x + m.m[1][0] * v.y + m.m[2][0] * v.z;
	float y = m.m[0][1] * v.x + m.m[1][1] * v.y + m.m[2][1] * v.z;
	float z = m.m[0][2] * v.x + m.m[1][2] * v.y + m.m[2][2] * v.z;

	return make_float3(x, y, z);
}

KDRAY_INLINE float3 transform_point(const float4x4& m, const float3& v)
{
	float x = m.m[0][0] * v.x + m.m[1][0] * v.y + m.m[2][0] * v.z + m.m[3][0];
	float y = m.m[0][1] * v.x + m.m[1][1] * v.y + m.m[2][1] * v.z + m.m[3][1];
	float z = m.m[0][2] * v.x + m.m[1][2] * v.y + m.m[2][2] * v.z + m.m[3][2];
	float w = m.m[0][3] * v.x + m.m[1][3] * v.y + m.m[2][3] * v.z + m.m[3][3];

	return make_float3(x, y, z) / w;
}

KDRAY_INLINE float4x4 inverse(const float4x4& m)
{
	float coef00 = m.m[2][2] * m.m[3][3] - m.m[3][2] * m.m[2][3];
	float coef02 = m.m[1][2] * m.m[3][3] - m.m[3][2] * m.m[1][3];
	float coef03 = m.m[1][2] * m.m[2][3] - m.m[2][2] * m.m[1][3];

	float coef04 = m.m[2][1] * m.m[3][3] - m.m[3][1] * m.m[2][3];
	float coef06 = m.m[1][1] * m.m[3][3] - m.m[3][1] * m.m[1][3];
	float coef07 = m.m[1][1] * m.m[2][3] - m.m[2][1] * m.m[1][3];

	float coef08 = m.m[2][1] * m.m[3][2] - m.m[3][1] * m.m[2][2];
	float coef10 = m.m[1][1] * m.m[3][2] - m.m[3][1] * m.m[1][2];
	float coef11 = m.m[1][1] * m.m[2][2] - m.m[2][1] * m.m[1][2];

	float coef12 = m.m[2][0] * m.m[3][3] - m.m[3][0] * m.m[2][3];
	float coef14 = m.m[1][0] * m.m[3][3] - m.m[3][0] * m.m[1][3];
	float coef15 = m.m[1][0] * m.m[2][3] - m.m[2][0] * m.m[1][3];

	float coef16 = m.m[2][0] * m.m[3][2] - m.m[3][0] * m.m[2][2];
	float coef18 = m.m[1][0] * m.m[3][2] - m.m[3][0] * m.m[1][2];
	float coef19 = m.m[1][0] * m.m[2][2] - m.m[2][0] * m.m[1][2];

	float coef20 = m.m[2][0] * m.m[3][1] - m.m[3][0] * m.m[2][1];
	float coef22 = m.m[1][0] * m.m[3][1] - m.m[3][0] * m.m[1][1];
	float coef23 = m.m[1][0] * m.m[2][1] - m.m[2][0] * m.m[1][1];

	float4 fac0 = make_float4(coef00, coef00, coef02, coef03);
	float4 fac1 = make_float4(coef04, coef04, coef06, coef07);
	float4 fac2 = make_float4(coef08, coef08, coef10, coef11);
	float4 fac3 = make_float4(coef12, coef12, coef14, coef15);
	float4 fac4 = make_float4(coef16, coef16, coef18, coef19);
	float4 fac5 = make_float4(coef20, coef20, coef22, coef23);

	float4 cec0 = make_float4(m.m[1][0], m.m[0][0], m.m[0][0], m.m[0][0]);
	float4 cec1 = make_float4(m.m[1][1], m.m[0][1], m.m[0][1], m.m[0][1]);
	float4 cec2 = make_float4(m.m[1][2], m.m[0][2], m.m[0][2], m.m[0][2]);
	float4 cec3 = make_float4(m.m[1][3], m.m[0][3], m.m[0][3], m.m[0][3]);

	float4 inv0 = cec1 * fac0 - cec2 * fac1 + cec3 * fac2;
	float4 inv1 = cec0 * fac0 - cec2 * fac3 + cec3 * fac4;
	float4 inv2 = cec0 * fac1 - cec1 * fac3 + cec3 * fac5;
	float4 inv3 = cec0 * fac2 - cec1 * fac4 + cec2 * fac5;

	float4x4 _m = make_float4x4(
		inv0.x, -inv0.y, inv0.z, -inv0.w,
		-inv1.x, inv1.y, -inv1.z, inv1.w,
		inv2.x, -inv2.y, inv2.z, -inv2.w,
		-inv3.x, inv3.y, -inv3.z, inv3.w);

	float4 row0 = make_float4(_m.m[0][0], _m.m[1][0], _m.m[2][0], _m.m[3][0]);
	float4 dot0 = m[0] * row0;
	float idet = (dot0.x + dot0.y) + (dot0.z + dot0.w);
	return _m * (1.0f / idet);
}
