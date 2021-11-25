#version 130
//#include "common.frag"

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

// forward declarations
float sceneSDF(vec3 p);

// OPERATIONS ----------------------------------------------------------

float opUnion(float d1, float d2)
{
    return min(d1, d2);
}

float opSubtraction(float d1, float d2)
{
	return max(-d1, d2);
}

float opIntersection(float d1, float d2)
{
	return max(d1, d2);
}

float opSmoothUnion(float d1, float d2, float k)
{
    float h = clamp(0.5 + 0.5 * (d2-d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

float opSmoothSubtraction(float d1, float d2, float k)
{
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}

float opSmoothIntersection(float d1, float d2, float k)
{
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

// shape blending
// d1 - object1, d2 - object2, a - factor [0, 1]
float sdf_blend(float d1, float d2, float a)
{
    return a * d1 + (1 - a) * d2;
}

// smoothing SDF union between two shapes
// polynomial smooth min 1 (k=0.1)
float smin(float a, float b, float k)
{
	float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
	return mix(b, a, h) - k * h * (1.0 - h);
}

// exponential smooth min (k=32)
float smin_exp(float a, float b, float k = 32)
{
    float res = exp(-k*a) + exp(-k*b);
    return -log(max(0.0001,res)) / k;
}

// TRANSFORMATION -----------------------------------------------------

// Rotate around a coordinate axis (i.e. in a plane perpendicular to that axis) by angle <a>.
// Read like this: R(p.xz, a) rotates "x towards z".
// This is fast if <a> is a compile-time constant and slower (but still practical) if not.
void pR(inout vec2 p, float a) {
	p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}

// Repeat space along one axis. Use like this to repeat along the x axis:
// <float cell = pMod1(p.x,5);> - using the return value is optional.
float pMod1(inout float p, float size) {
	float halfsize = size*0.5;
	float c = floor((p + halfsize)/size);
	p = mod(p + halfsize, size) - halfsize;
	return c;
}

// Repeat in two dimensions
vec2 pMod2(inout vec2 p, vec2 size) {
	vec2 c = floor((p + size*0.5)/size);
	p = mod(p + size*0.5,size) - size*0.5;
	return c;
}

// Mirror at an axis-aligned plane which is at a specified distance <dist> from the origin.
float pMirror (inout float p, float dist) {
	float s = (p<0)?-1:1;
	p = abs(p)-dist;
	return s;
}

// Reflect space at a plane
float pReflect(inout vec3 p, vec3 planeNormal, float offset) {
	float t = dot(p, planeNormal)+offset;
	if (t < 0) {
		p = p - (2*t)*planeNormal;
	}
	return (t<0)?-1:1;
}

// SHAPES -------------------------------------------------------------
// return: distance from the surface to the sample point (p)

float plane(vec3 p)
{
	return p.y;
}

// p: plane origin (position), n.xyz: plane surface normal, p.w: plane's distance from origin (along its normal)
float sdPlane(vec3 p, vec4 n)
{
    return dot(p, n.xyz) + n.w;
}

// s - position, radius
float sphere(vec4 s, vec3 p)
{
	return length(p - s.xyz) - s.w;
}

float cube(vec4 s, vec3 p)
{
	vec3 q = abs(p - s.xyz) - s.w;
	return length(max(q, 0)) + min(max(q.x, max(q.y, q.z)), 0);
}

float cylinder(const in vec3 p,  float r )
{
	return length(p.xy)-r;
}

float cone( vec3 p, vec2 c )
{    // c must be normalized
    float q = length(p.xy);
    return dot(c,vec2(q,p.z));
}

float torus( vec3 p, vec2 t )
{
  vec2 q = vec2(length(p.xy)-t.x, p.z);
  return length(q)-t.y;
}


// LIGHTING ------------------------------------------------------------

// http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    // bounding volume
    float tp = (0.8-ro.y)/rd.y; if( tp>0.0 ) tmax = min( tmax, tp );

    float res = 1.0;
    float t = mint;
    for( int i=0; i<24; i++ )
    {
		float h = sceneSDF( ro + rd*t );
        float s = clamp(8.0*h/t,0.0,1.0);
        res = min( res, s*s*(3.0-2.0*s) );
        t += clamp( h, 0.02, 0.2 );
        if( res<0.004 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

//float shadow(vec3 pt)
//{
//  vec3 lightDir = normalize(lightPos - pt);
//  float kd = 1;
//  int step = 0;
//  for (float t = 0.1; 
//      t < length(lightPos - pt) 
//      && step < renderDepth && kd > 0.001; ) {
//    float d = abs(getSDF(pt + t * lightDir));
//    if (d < 0.001) {
//      kd = 0;
//    } else {
//      kd = min(kd, 16 * d / t);
//    }
//    t += d;
//    step++;
//  }
//  return kd;
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

float calcAO( in vec3 pos, in vec3 nor )
{
	float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ )
    {
        float h = 0.01 + 0.12*float(i)/4.0;
        float d = sceneSDF( pos + h*nor );
        occ += (h-d)*sca;
        sca *= 0.95;
        if( occ>0.35 ) break;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 ) * (0.5+0.5*nor.y);
}

const float _AOSteps = 5;
const float _AOStep = 0.1;
const float _AOStepSize = 1.0;

// good ambient occlusion
float ambientOcclusion(vec3 pos, vec3 normal)
{
    float sum = 0;
    for (int i = 0; i < _AOSteps; i ++)
    {
        vec3 p = pos + normal * (i+1) * _AOStepSize;
        sum    += sceneSDF(p);
    }
    return sum / (_AOStep * _AOStepSize);
}

// ambient occlusion (realistic, exponential decay)
float ambientOcclusionReal(vec3 pos, vec3 normal)
{
    float sum    = 0;
    float maxSum = 0;
    for (int i = 0; i < _AOSteps; i ++)
    {
        vec3 p = pos + normal * (i+1) * _AOStepSize;
        sum    += 1. / pow(2., i) * sceneSDF(p);
        maxSum += 1. / pow(2., i) * (i+1) * _AOStepSize;
    }
    return sum / maxSum;
}

// POST-PROCESSING -----------------------------------------------------

vec3 gammaCorrection(vec3 color)
{
	return pow(color, vec3(0.4545)); // 0.4545 == 1 / 2.2
}

// ---------------------------------------------------------------------




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
	
	// fast ambient occlusion
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
