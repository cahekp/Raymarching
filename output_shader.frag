//#extension GL_ARB_shading_language_include : require
#version 130

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform vec3 u_pos;
uniform float u_time;
uniform sampler2D u_sample;
uniform float u_sample_part;
uniform vec2 u_seed1;
uniform vec2 u_seed2;

const float ZNEAR = 0;
const float ZFAR = 50;
const int MAX_MARCHING_STEPS = 500;

float smin(float a, float b, float k)
{
	float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
	return mix(b, a, h) - k * h * (1.0 - h);
}

float plane(vec3 p)
{
	return p.y;
}

float sphere(vec4 s, vec3 p)
{
	return length(p - s.xyz) - s.w;
}

float cube(vec4 s, vec3 p)
{
	vec3 q = abs(p - s.xyz) - s.w;
	return length(max(q, 0)) + min(max(q.x, max(q.y, q.z)), 0);
}

// return distance to nearest intersection
float sceneSDF(vec3 p)
{
	float dist1 = sphere(vec4(0, 0, 2, 1), p);
	float dist2 = cube(vec4(0, 0, 0, 1), p);
	float dist3 = plane(p);
	return smin(smin(dist1, dist2, 0.5), dist3, 0.5);
}

//vec3 getNormal(vec3 p)
//{
//	float d = sceneSDF(p);
//	vec2 e = vec2(0.001, 0);
//	vec3 n = d - vec3(sceneSDF(p - e.xyy), sceneSDF(p - e.yxy), sceneSDF(p - e.yyx));
//	return normalize(n);
//}

vec3 getNormal(vec3 p)
{
	float eps = 0.001;
	return normalize(vec3(
		sceneSDF(vec3(p.x + eps, p.y, p.z)) - sceneSDF(vec3(p.x - eps, p.y, p.z)),
		sceneSDF(vec3(p.x, p.y + eps, p.z)) - sceneSDF(vec3(p.x, p.y - eps, p.z)),
		sceneSDF(vec3(p.x, p.y, p.z  + eps)) - sceneSDF(vec3(p.x, p.y, p.z - eps))
	));
}

float raymarchLight(vec3 ro, vec3 rd)
{
	float dO = 0;
	float md = 1;
	for (int i = 0; i < 20; i++)
	{
		vec3 p = ro + rd * dO;
		float dS = sceneSDF(p);
		md = min(md, dS);
		dO += dS;
		if(dO > 50 || dS < 0.1) break;
	}
	return md;
}

vec4 getLight(vec3 p, vec3 ro, int i, vec3 lightPos)
{
	vec3 l = normalize(lightPos - p);
	vec3 n = getNormal(p);
	float dif = clamp(dot(n, l) * 0.5 + 0.5, 0, 1);
	float d = raymarchLight(p + n * 0.1 * 10, l);
	d += 1;
	d = clamp(d, 0, 1);
	dif *= d;
	vec4 col = vec4(dif, dif, dif, 1);
	float occ = (float(i) / MAX_MARCHING_STEPS * 2);
	occ = 1 - occ;
	occ *= occ;
	col.rgb *= occ;
	float fog = distance(p, ro);
	fog /= ZFAR;
	fog = clamp(fog, 0, 1);
	fog *= fog;
	col.rgb = col.rgb * (1 - fog) + 0.28 * fog;
	return col;
}

// ro - ray origin
// rd - ray direction
vec4 raymarch(vec3 ro, vec3 rd)
{
	float depth = 0 /* znear */;
	for (int i = 0; i < MAX_MARCHING_STEPS; i++)
	{
		// get a distance to the nearest scene's surface
		float dist = sceneSDF(ro + rd * depth);

		// found intersection?
		if (dist < 0.001)
		{
			vec3 p = ro + rd * depth;
			return getLight(p, ro, i, vec3(0, 50, 0));
		}
		
		// move along the view ray
		depth += dist;

		// reached zfar?
		if (depth >= ZFAR)
			return vec4(0);

	}
	return vec4(0);
}

mat2 rot(float a) {
	float s = sin(a);
	float c = cos(a);
	return mat2(c, -s, s, c);
}

void main() {
	vec2 uv = (gl_TexCoord[0].xy - 0.5) * u_resolution / u_resolution.y;
	vec3 rayOrigin = u_pos;
	vec3 rayDirection = normalize(vec3(uv.x, -uv.y, -1.0));
	rayDirection.yz *= rot(-u_mouse.y);
	rayDirection.xz *= rot(u_mouse.x);
	vec3 col = vec3(raymarch(rayOrigin, rayDirection));
	gl_FragColor = vec4(col, 1.0);
}
