#version 130
#extension GL_ARB_gpu_shader5 : enable // enable inverse() func
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
const int MAX_MARCHING_STEPS = 128;

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

float rounding(in float d, in float h = 0.1)
{
    return d - h;
}

// TRANSFORMATION: FAST, BUT HARD TO USE (SOMETIMES) ------------------

void translatePoint(inout vec3 p, vec3 offset)
{
	p = p - offset;
}

// Rotate around a coordinate axis (i.e. in a plane perpendicular to that axis) by angle <a>.
// Read like this: rotatePoint(p.xz, a) rotates "x towards z".
// This is fast if <a> is a compile-time constant and slower (but still practical) if not.
void rotatePoint(inout vec2 p, float a) {
	p = cos(a) * p + sin(a) * vec2(p.y, -p.x);
}

vec3 rotatePointX(vec3 p, float a)
{
	p.yz = cos(a) * p.yz + sin(a) * vec2(p.z, -p.y);
	return p;
}

vec3 rotatePointY(vec3 p, float a)
{
	p.xz = cos(a) * p.xz + sin(a) * vec2(p.z, -p.x);
	return p;
}

vec3 rotatePointZ(vec3 p, float a)
{
	p.xy = cos(a) * p.xy + sin(a) * vec2(p.y, -p.x);
	return p;
}

#define scaleSDF(func, samplePoint, scaleFactor) (func(samplePoint / scaleFactor) * scaleFactor)
#define scaleSDF3(func, samplePoint, s_x, s_y, s_z) (func(samplePoint / vec3(s_x, s_y, s_z)) * min(s_x, min(s_y, s_z)))

// TRANSFORMATION: SLOW, BUT EASY TO USE ------------------------------

mat4 rotationX(in float angle_deg)
{
	float angle_rad = radians(angle_deg);
    float c = cos(angle_rad);
    float s = sin(angle_rad);
    return mat4(
        vec4(1, 0, 0, 0),
        vec4(0, c, -s, 0),
        vec4(0, s, c, 0),
        vec4(0, 0, 0, 1)
    );
}

mat4 rotationY(in float angle_deg)
{
	float angle_rad = radians(angle_deg);
    float c = cos(angle_rad);
    float s = sin(angle_rad);
    return mat4(
        vec4(c, 0, s, 0),
        vec4(0, 1, 0, 0),
        vec4(-s, 0, c, 0),
        vec4(0, 0, 0, 1)
    );
}

mat4 rotationZ(in float angle_deg)
{
	float angle_rad = radians(angle_deg);
    float c = cos(angle_rad);
    float s = sin(angle_rad);
    return mat4(
        vec4(c, -s, 0, 0),
        vec4(s, c, 0, 0),
        vec4(0, 0, 1, 0),
        vec4(0, 0, 0, 1)
    );
}

// this function uses inverse!
//vec3 transform(vec3 sample_point, vec3 pos, vec3 rot, vec3 scale)
//{
//	mat4 s = mat4(
//      vec4(scale.x, 0, 0, 0),
//      vec4(0, scale.y, 0, 0),
//      vec4(0, 0, scale.z, 0),
//      vec4(0, 0, 0, 1));
//
//	mat4 t = mat4(
//      vec4(1, 0, 0, pos.x),
//      vec4(0, 1, 0, pos.y),
//      vec4(0, 0, 1, pos.z),
//      vec4(0, 0, 0, 1));
//	  
//	return (vec4(sample_point, 1.0) *
//		inverse(s * rotationZ(rot.z) * rotationX(rot.x) * rotationY(rot.y) * t)).xyz;
//}

vec3 transform(in vec3 sample_point, in vec3 pos, in vec3 rot, in vec3 scale)
{
	mat4 s = mat4(
      vec4(1/scale.x, 0, 0, 0),
      vec4(0, 1/scale.y, 0, 0),
      vec4(0, 0, 1/scale.z, 0),
      vec4(0, 0, 0, 1));

	mat4 r_y = rotationY(-rot.y);
	mat4 r_x = rotationX(-rot.x);
	mat4 r_z = rotationZ(-rot.z);

	mat4 t = mat4(
      vec4(1, 0, 0, -pos.x),
      vec4(0, 1, 0, -pos.y),
      vec4(0, 0, 1, -pos.z),
      vec4(0, 0, 0, 1));
	  
	return (vec4(sample_point, 1.0) * t * r_y * r_x * r_z * s).xyz;
}

vec3 transformTR(in vec3 sample_point, in vec3 pos, in vec3 rot)
{
	mat4 r_y = rotationY(-rot.y);
	mat4 r_x = rotationX(-rot.x);
	mat4 r_z = rotationZ(-rot.z);

	mat4 t = mat4(
      vec4(1, 0, 0, -pos.x),
      vec4(0, 1, 0, -pos.y),
      vec4(0, 0, 1, -pos.z),
      vec4(0, 0, 0, 1));
	  
	return (vec4(sample_point, 1.0) * t * r_y * r_x * r_z).xyz;
}

vec3 transformTRX(in vec3 sample_point, in vec3 pos, in float rot_x)
{
	mat4 r_x = rotationX(-rot_x);

	mat4 t = mat4(
      vec4(1, 0, 0, -pos.x),
      vec4(0, 1, 0, -pos.y),
      vec4(0, 0, 1, -pos.z),
      vec4(0, 0, 0, 1));
	  
	return (vec4(sample_point, 1.0) * t * r_x).xyz;
}

vec3 transformTRY(in vec3 sample_point, in vec3 pos, in float rot_y)
{
	mat4 r_y = rotationY(-rot_y);

	mat4 t = mat4(
      vec4(1, 0, 0, -pos.x),
      vec4(0, 1, 0, -pos.y),
      vec4(0, 0, 1, -pos.z),
      vec4(0, 0, 0, 1));
	  
	return (vec4(sample_point, 1.0) * t * r_y).xyz;
}

vec3 transformTRZ(in vec3 sample_point, in vec3 pos, in float rot_z)
{
	mat4 r_z = rotationZ(-rot_z);

	mat4 t = mat4(
      vec4(1, 0, 0, -pos.x),
      vec4(0, 1, 0, -pos.y),
      vec4(0, 0, 1, -pos.z),
      vec4(0, 0, 0, 1));
	  
	return (vec4(sample_point, 1.0) * t * r_z).xyz;
}

vec3 transformTRXS(in vec3 sample_point, in vec3 pos, in float rot_x, in vec3 scale)
{
	mat4 s = mat4(
      vec4(1/scale.x, 0, 0, 0),
      vec4(0, 1/scale.y, 0, 0),
      vec4(0, 0, 1/scale.z, 0),
      vec4(0, 0, 0, 1));

	mat4 r_x = rotationX(-rot_x);

	mat4 t = mat4(
      vec4(1, 0, 0, -pos.x),
      vec4(0, 1, 0, -pos.y),
      vec4(0, 0, 1, -pos.z),
      vec4(0, 0, 0, 1));
	  
	return (vec4(sample_point, 1.0) * t * r_x * s).xyz;
}

vec3 transformTRYS(in vec3 sample_point, in vec3 pos, in float rot_y, in vec3 scale)
{
	mat4 s = mat4(
      vec4(1/scale.x, 0, 0, 0),
      vec4(0, 1/scale.y, 0, 0),
      vec4(0, 0, 1/scale.z, 0),
      vec4(0, 0, 0, 1));

	mat4 r_y = rotationY(-rot_y);

	mat4 t = mat4(
      vec4(1, 0, 0, -pos.x),
      vec4(0, 1, 0, -pos.y),
      vec4(0, 0, 1, -pos.z),
      vec4(0, 0, 0, 1));
	  
	return (vec4(sample_point, 1.0) * t * r_y * s).xyz;
}

vec3 transformTRZS(in vec3 sample_point, in vec3 pos, in float rot_z, in vec3 scale)
{
	mat4 s = mat4(
      vec4(1/scale.x, 0, 0, 0),
      vec4(0, 1/scale.y, 0, 0),
      vec4(0, 0, 1/scale.z, 0),
      vec4(0, 0, 0, 1));

	mat4 r_z = rotationZ(-rot_z);

	mat4 t = mat4(
      vec4(1, 0, 0, -pos.x),
      vec4(0, 1, 0, -pos.y),
      vec4(0, 0, 1, -pos.z),
      vec4(0, 0, 0, 1));
	  
	return (vec4(sample_point, 1.0) * t * r_z * s).xyz;
}

vec3 transformTRS1(in vec3 sample_point, in vec3 pos, in vec3 rot, in float scale)
{
	mat4 r_y = rotationY(-rot.y);
	mat4 r_x = rotationX(-rot.x);
	mat4 r_z = rotationZ(-rot.z);

	mat4 t = mat4(
      vec4(1, 0, 0, -pos.x),
      vec4(0, 1, 0, -pos.y),
      vec4(0, 0, 1, -pos.z),
      vec4(0, 0, 0, 1));
	  
	return (vec4(sample_point, 1.0) * t * r_y * r_x * r_z).xyz / scale;
}

vec3 transformTRXS1(in vec3 sample_point, in vec3 pos, in float rot_x, in float scale)
{
	mat4 r_x = rotationX(-rot_x);

	mat4 t = mat4(
      vec4(1, 0, 0, -pos.x),
      vec4(0, 1, 0, -pos.y),
      vec4(0, 0, 1, -pos.z),
      vec4(0, 0, 0, 1));
	  
	return (vec4(sample_point, 1.0) * t * r_x).xyz / scale;
}

vec3 transformTRYS1(in vec3 sample_point, in vec3 pos, in float rot_y, in float scale)
{
	mat4 r_y = rotationY(-rot_y);

	mat4 t = mat4(
      vec4(1, 0, 0, -pos.x),
      vec4(0, 1, 0, -pos.y),
      vec4(0, 0, 1, -pos.z),
      vec4(0, 0, 0, 1));
	  
	return (vec4(sample_point, 1.0) * t * r_y).xyz / scale;
}

vec3 transformTRZS1(in vec3 sample_point, in vec3 pos, in float rot_z, in float scale)
{
	mat4 r_z = rotationZ(-rot_z);

	mat4 t = mat4(
      vec4(1, 0, 0, -pos.x),
      vec4(0, 1, 0, -pos.y),
      vec4(0, 0, 1, -pos.z),
      vec4(0, 0, 0, 1));
	  
	return (vec4(sample_point, 1.0) * t * r_z).xyz / scale;
}

vec3 transformR(in vec3 sample_point, in vec3 rot)
{
	mat4 r_y = rotationY(-rot.y);
	mat4 r_x = rotationX(-rot.x);
	mat4 r_z = rotationZ(-rot.z);
	
	return (vec4(sample_point, 1.0) * r_y * r_x * r_z).xyz;
}

vec3 transformRX(in vec3 sample_point, in float rot_x)
{
	mat4 r_x = rotationX(-rot_x);
	
	return (vec4(sample_point, 1.0) * r_x).xyz;
}

vec3 transformRY(in vec3 sample_point, in float rot_y)
{
	mat4 r_y = rotationY(-rot_y);
	
	return (vec4(sample_point, 1.0) * r_y).xyz;
}

vec3 transformRZ(in vec3 sample_point, in float rot_z)
{
	mat4 r_z = rotationZ(-rot_z);

	return (vec4(sample_point, 1.0) * r_z).xyz;
}

vec3 transformRXS(in vec3 sample_point, in float rot_x, in vec3 scale)
{
	mat4 s = mat4(
      vec4(1/scale.x, 0, 0, 0),
      vec4(0, 1/scale.y, 0, 0),
      vec4(0, 0, 1/scale.z, 0),
      vec4(0, 0, 0, 1));

	mat4 r_x = rotationX(-rot_x);
 
	return (vec4(sample_point, 1.0) * r_x * s).xyz;
}

vec3 transformRYS(in vec3 sample_point, in float rot_y, in vec3 scale)
{
	mat4 s = mat4(
      vec4(1/scale.x, 0, 0, 0),
      vec4(0, 1/scale.y, 0, 0),
      vec4(0, 0, 1/scale.z, 0),
      vec4(0, 0, 0, 1));

	mat4 r_y = rotationY(-rot_y);

	return (vec4(sample_point, 1.0) * r_y * s).xyz;
}

vec3 transformRZS(in vec3 sample_point, in float rot_z, in vec3 scale)
{
	mat4 s = mat4(
      vec4(1/scale.x, 0, 0, 0),
      vec4(0, 1/scale.y, 0, 0),
      vec4(0, 0, 1/scale.z, 0),
      vec4(0, 0, 0, 1));

	mat4 r_z = rotationZ(-rot_z);

	return (vec4(sample_point, 1.0) * r_z * s).xyz;
}

vec3 transformRS1(in vec3 sample_point, in vec3 rot, in float scale)
{
	mat4 r_y = rotationY(-rot.y);
	mat4 r_x = rotationX(-rot.x);
	mat4 r_z = rotationZ(-rot.z);

	return (vec4(sample_point, 1.0) * r_y * r_x * r_z).xyz / scale;
}

vec3 transformRXS1(in vec3 sample_point, in float rot_x, in float scale)
{
	mat4 r_x = rotationX(-rot_x);

	return (vec4(sample_point, 1.0) * r_x).xyz / scale;
}

vec3 transformRYS1(in vec3 sample_point, in float rot_y, in float scale)
{
	mat4 r_y = rotationY(-rot_y);

	return (vec4(sample_point, 1.0) * r_y).xyz / scale;
}

vec3 transformRZS1(in vec3 sample_point, in float rot_z, in float scale)
{
	mat4 r_z = rotationZ(-rot_z);

	return (vec4(sample_point, 1.0) * r_z).xyz / scale;
}


// --------------------------------------------------------------------

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

// SHAPES: SIMPLE -----------------------------------------------------
// (or, more correctly, "distance estimators")
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

float sdBox( vec3 p, vec3 b )
{
    vec3  di = abs(p) - b;
    float mc = max(di.x,max(di.y,di.z));
    return min(mc,length(max(di,0.0)));
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

// SHAPES: FRACTALS ---------------------------------------------------
// return: distance from the surface to the sample point (p)

float mandelbulb( in vec3 p, out vec4 resColor )
{
    vec3 w = p;
    float m = dot(w,w);

    vec4 trap = vec4(abs(w),m);
	float dz = 1.0;
    
	for( int i=0; i<4; i++ )
    {
		dz = 8.0*pow(m,3.5)*dz + 1.0;
      
        // z = z^8+z
        float r = length(w);
        float b = 8.0*acos( w.y/r);
        float a = 8.0*atan( w.x, w.z );
        w = p + pow(r,8.0) * vec3( sin(b)*sin(a), cos(b), sin(b)*cos(a) );
        
        trap = min( trap, vec4(abs(w),m) );

        m = dot(w,w);
		if( m > 256.0 )
            break;
    }

    resColor = vec4(m,trap.yzw);

    // distance estimation (through the Hubbard-Douady potential)
    return 0.25*log(m)*sqrt(m)/dz;
}

// menger sponge
vec3 mengersponge(in vec3 p)
{
   float d = sdBox(p,vec3(1.0));
   vec3 res = vec3(d, 1.0, 0.0);

   float s = 1.0;
   for( int m=0; m<3; m++ )
   {
      vec3 a = mod( p*s, 2.0 )-1.0;
      s *= 3.0;
      vec3 r = abs(1.0 - 3.0*abs(a));

      float da = max(r.x,r.y);
      float db = max(r.y,r.z);
      float dc = max(r.z,r.x);
      float c = (min(da,min(db,dc))-1.0)/s;

      if( c>d )
      {
          d = c;
          res = vec3( d, 0.2*da*db*dc, (1.0+float(m))/4.0);
       }
   }

   return res;
}

// LIGHTING ------------------------------------------------------------

// find normal vector of the scene surface
// classic technique - forward and central differences (6 sceneSDF)
vec3 getNormal(in vec3 p)
{
	float eps = 0.001;
	return normalize(vec3(
		sceneSDF(vec3(p.x + eps, p.y, p.z)) - sceneSDF(vec3(p.x - eps, p.y, p.z)),
		sceneSDF(vec3(p.x, p.y + eps, p.z)) - sceneSDF(vec3(p.x, p.y - eps, p.z)),
		sceneSDF(vec3(p.x, p.y, p.z  + eps)) - sceneSDF(vec3(p.x, p.y, p.z - eps))
	));
}

// find normal vector of the scene surface
// tetrahedron technique (4 sceneSDF)
vec3 getNormalFast(in vec3 p)
{
	const float h = 0.001;
	const vec3 k0 = vec3(1.0, -1.0, -1.0);
	const vec3 k1 = vec3(-1.0, -1.0, 1.0);
	const vec3 k2 = vec3(-1.0, 1.0, -1.0);
	const vec3 k3 = vec3(1.0, 1.0, 1.0);
    return normalize(k0 * sceneSDF(p + k0 * h) +
                     k1 * sceneSDF(p + k1 * h) +
                     k2 * sceneSDF(p + k2 * h) +
                     k3 * sceneSDF(p + k3 * h));
}

/**
 * Lighting contribution of a single point light source via Phong illumination.
 * 
 * The vec3 returned is the RGB color of the light's contribution.
 *
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 * lightPos: the position of the light
 * lightIntensity: color/intensity of the light
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 phongContribForLight(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                          vec3 lightPos, vec3 lightIntensity)
{
    vec3 N = getNormalFast(p);
    vec3 L = normalize(lightPos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));
    
    float dotLN = dot(L, N);
    float dotRV = dot(R, V);
    
    if (dotLN < 0.0)
	{
        // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    } 
    
    if (dotRV < 0.0)
	{
        // Light reflection in opposite direction as viewer, apply only diffuse
        // component
        return lightIntensity * (k_d * dotLN);
    }
    return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}

// example of use
//vec3 phongIllumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye) {
//    const vec3 ambientLight = 0.5 * vec3(1.0, 1.0, 1.0);
//    vec3 color = ambientLight * k_a;
//    
//    vec3 light1Pos = vec3(4.0 * sin(iTime),
//                          2.0,
//                          4.0 * cos(iTime));
//    vec3 light1Intensity = vec3(0.4, 0.4, 0.4);
//    
//    color += phongContribForLight(k_d, k_s, alpha, p, eye,
//                                  light1Pos,
//                                  light1Intensity);
//    
//    vec3 light2Pos = vec3(2.0 * sin(0.37 * iTime),
//                          2.0 * cos(0.37 * iTime),
//                          2.0);
//    vec3 light2Intensity = vec3(0.4, 0.4, 0.4);
//    
//    color += phongContribForLight(k_d, k_s, alpha, p, eye,
//                                  light2Pos,
//                                  light2Intensity);    
//    return color;
//}
// vec3 K_a = vec3(0.2, 0.2, 0.2);
// vec3 K_d = vec3(0.7, 0.2, 0.2);
// vec3 K_s = vec3(1.0, 1.0, 1.0);
// float shininess = 10.0;

// sharp shadows
// https://iquilezles.org/www/articles/rmshadows/rmshadows.htm
// ro + rd - ray from a sample point to the light source
// mint - shadow bias (to prevent "shadow acne" effect)
// maxt - distance to the light source
// return 0 if there is intersection ("ro" in the shadow)
// return 1 if sample point "ro" is fully illuminated
float shadow(in vec3 ro, in vec3 rd, float mint, float maxt)
{
    for (float t = mint; t < maxt;)
    {
        float d = sceneSDF(ro + rd * t);
        if (d < 0.001)
            return 0.0;
        t += d;
    }
    return 1.0;
}

// soft shadows
// k - sharping factor (k = 2 - soft shadows, 128 - sharp shadows)
float softshadow(in vec3 ro, in vec3 rd, float mint, float maxt, float k = 2)
{
    float res = 1.0;
    for (float t = mint; t < maxt;)
    {
        float d = sceneSDF(ro + rd * t);
        if (d < 0.001)
            return 0.0;
        res = min(res, k * d / t);
		t += d;
    }
    return res;
}

// soft shadows (improved by Sebastian Aaltonen at GDC)
float softshadow2(in vec3 ro, in vec3 rd, float mint, float maxt, float k = 2)
{
    float res = 1.0;
    float ph = 1e20;
    for (float t = mint; t < maxt;)
    {
        float h = sceneSDF(ro + rd * t);
        if (h < 0.001)
            return 0.0;
        float y = h * h / (2.0 * ph);
        float d = sqrt(h * h - y * y);
        res = min(res, k * d / max(0.0, t - y));
        ph = h;
		
		// simple fix to prevent color banding of the shadow
		// (increase quality, decrease performance)
		t += h * 0.1 + 0.001;
		// original method (produces color banding sometime) is:
        // t += h;
    }
    return res;
}

float ambientOcclusion( in vec3 pos, in vec3 nor )
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


// the more value, the better the AO quality
const float _AOSteps = 4;
// the smaller the step, the darker the small details
const float _AOStepSize = 0.2;

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

vec3 gammaCorrection(in vec3 color)
{
	return pow(color, vec3(0.4545)); // 0.4545 == 1 / 2.2
}

// SCENE ---------------------------------------------------------------

/**
 * Signed distance function describing the scene.
 * 
 * Absolute value of the return value indicates the distance to the surface.
 * Sign indicates whether the point is inside or outside the surface,
 * negative indicating inside.
 */
// distance field
float sceneSDF(vec3 p)
{
	vec4 c = vec4(1);
	//float dist0 = mandelbulb(transformTRS1(p, vec3(0, 2, 0), vec3(u_time * 0.5, 0, 45), 1.5), c) * 1.5;
	
	
	//float dist0 = mandelbulb(transformRS1(p - vec3(0, 2, 0), vec3(180, u_time * 2, 0), 1.5), c) * 1.5;
	float dist0 = mengersponge(transformR(p - vec3(0, 3, 0), vec3(180, u_time * 2, 0))).x;
	
	float dist1 = sphere(vec4(0, 0, 2, 1), p);
	//float dist1 = sphere(vec4(0, 1.5, 0, 1), p / 1.2) * 1.2;
	float dist2 = cube(vec4(0, 0, 0, 1), p);
	float dist3 = plane(p);
	return smin(dist0, smin(smin(dist1, dist2, 0.5), dist3, 0.5), 0.33);
}

// ---------------------------------------------------------------------

// DELETE?
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

// find intersection using raymarching SDF scene
// ro - ray origin
// rd - ray direction
// return: intersection point
vec3 castRay(in vec3 ro, in vec3 rd)
{
	float depth = ZNEAR;
	vec3 p = ro + rd * depth;
	for (int i = 0; i < MAX_MARCHING_STEPS; i++)
	{
		// get a distance to the nearest scene's surface
		float dist = sceneSDF(p);

		// found intersection?
		if (dist < 0.001)
			return p;
		
		// move along the view ray
		depth += dist;
		p = ro + rd * depth;

		// reached zfar?
		if (depth >= ZFAR)
			return ro + rd * ZFAR;

	}
	return p;
}

// find intersection inside a solid geometry using raymarching SDF scene
vec3 castRayInside(in vec3 ro, in vec3 rd)
{
	float depth = ZNEAR;
	vec3 p = ro + rd * depth;
	for (int i = 0; i < MAX_MARCHING_STEPS; i++)
	{
		// get a distance to the nearest scene's surface
		float dist = -sceneSDF(p);

		// found intersection?
		if (dist < 0.001)
			return p;
		
		// move along the view ray
		depth += dist;
		p = ro + rd * depth;

		// reached zfar?
		if (depth >= ZFAR)
			return ro + rd * ZFAR;

	}
	return p;
}

// default, diffuse material
vec3 getColor(in vec3 p)
{
	return vec3(1.0, 1.0, 1.0);
}

// p - point on glass surface
// n - glas surface normal
// rd - view direction vector
vec3 getColorReflect(in vec3 p, in vec3 n, in vec3 rd)
{
	vec3 reflect_dir = reflect(rd, n);
    vec3 pr = castRay(p + reflect_dir * 0.01 /*must be more than eps*/, reflect_dir);
    vec3 nr = getNormalFast(pr);
    
	// return simple depth texture + material color
    vec3 c = getColor(pr);
	c = c * clamp(length(pr - p) / 3.0, 0.0, 1.0);
	
	return c;
}

vec3 getColorRefract(in vec3 p, in vec3 n, in vec3 rd)
{
	const float REFRACTION_IDX = 1.52; // glass refraction index
	
	// cast ray inside the object
	vec3 refract_dir = refract(rd, n, 1.0 / REFRACTION_IDX);
    vec3 pr = castRayInside(p + refract_dir * 0.01, refract_dir);
    vec3 nr = -getNormalFast(pr);

	// cast ray outside the object
	refract_dir = refract(refract_dir, nr, 1.0 / REFRACTION_IDX);
    pr = castRay(pr + refract_dir * 0.01, refract_dir);
    nr = getNormalFast(pr);
    
	// return simple depth texture + material color
    vec3 c = getColor(pr);
	c = c * clamp(length(pr - p) / 3.0, 0.0, 1.0);
	return c;
}

// DELETE?
vec3 shade(vec3 o, vec3 p, vec3 n, vec3 lo)
{
    vec3 l = normalize(lo - p);
	
	// add light
	float dif = clamp(dot(n, l) * 0.5 + 0.5, 0, 1);
	float d = raymarchLight(p + n * 0.1 * 10, l);
	d += 1;
	d = clamp(d, 0, 1);
	dif *= d;
	vec3 col = vec3(dif, dif, dif);
	
    float sh = softshadow(p, l, 0.1, length(lo - p), 4.0);
	vec3 cshadow = pow( vec3(sh), vec3(1.0, 1.2, 1.5) );
	
    return mix(cshadow, col, sh);
}

vec3 applyFog(in vec3 color, in vec3 ro, in vec3 p, in vec3 fog_color)
{
	float fog_amount = clamp(distance(p, ro) / ZFAR, 0.0, 1.0);
	fog_amount *= fog_amount;
	return mix(color, fog_color, fog_amount);
}

// be/bi - fallof parameters for the extinction and inscattering
vec3 applyScattering(in vec3 color, in vec3 ro, in vec3 p, in vec3 fog_color,
	in vec3 be = vec3(2.0), in vec3 bi = vec3(2.0))
{
	float d = 1.0 - clamp(distance(p, ro) / ZFAR, 0.0, 1.0);

	// ext_color = extinction: absortion of light due to scattering
	// ins_color = inscattering
	vec3 ext_color = vec3(exp(-d * be.x), exp(-d * be.y), exp(-d * be.z));
	vec3 ins_color = vec3(exp(-d * bi.x), exp(-d * bi.y), exp(-d * bi.z));
	return color * (1.0 - ext_color) + fog_color * ins_color;
}

vec3 tonemap(in vec3 color)
{
    vec3 col = color*2.0/(1.0+color);
	col = pow(col, vec3(0.4545)); // gamma correction
    col = pow(col,vec3(0.85,0.97,1.0));
    col = col*0.5 + 0.5*col*col*(3.0-2.0*col);
	return col;
}

// INCORRECT!
// use this function after gamma correction pass
vec3 brightness(in vec3 color, float value = 0.9)
{
	return color * value;
}

// INCORRECT!
// use this function after gamma correction pass
vec3 saturation(in vec3 color, float value = 0.85)
{
	return mix(vec3(length(color)),color,value);
}

vec3 contrast(in vec3 color)
{
	return smoothstep(0.15, 1.1, color);
}

vec3 vignette(in vec3 color, in vec2 uv, float amount = 0.1)
{
	return color * (0.5 + 0.5 * pow(16.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), amount));
}

vec3 interlacedScan(vec3 color, vec2 uv)
{
	uv.y *= u_resolution.y / 240.0;
	color.r *= (0.5 + abs(0.5 - mod(uv.y        , 0.021) / 0.021) * 0.5) * 1.5;
	color.g *= (0.5 + abs(0.5 - mod(uv.y + 0.007, 0.021) / 0.021) * 0.5) * 1.5;
	color.b *= (0.5 + abs(0.5 - mod(uv.y + 0.014, 0.021) / 0.021) * 0.5) * 1.5;
	return color;
}

// should be in another shader (post shader)
vec3 chromaticAberration(sampler2D t, vec2 UV)
{
	vec2 uv = 1.0 - 2.0 * UV;
	vec3 c = vec3(0);
	float rf = 1.0;
	float gf = 1.0;
    float bf = 1.0;
	float f = 1.0 / 8.0;
	for(int i = 0; i < 8; ++i){
		c.r += f*texture(t, 0.5-0.5*(uv*rf) ).r;
		c.g += f*texture(t, 0.5-0.5*(uv*gf) ).g;
		c.b += f*texture(t, 0.5-0.5*(uv*bf) ).b;
		rf *= 0.9972;
		gf *= 0.998;
        bf /= 0.9988;
		c = clamp(c,0.0, 1.0);
	}
	return c;
}

// ro - ray origin
// rd - ray direction
// return: color of the intersection
vec3 render(in vec3 ro, in vec3 rd)
{
	// find intersection point and its normal
    vec3 p = castRay(ro, rd);
    vec3 n = getNormalFast(p);
	
	// find color of the intersection point
    vec3 c = getColorReflect(p, n, rd);
	
	// add lights and shadows
	vec3 lightPos = vec3(20, 50, 0);
	vec3 lightDir = normalize(lightPos - p);
	float occ = ambientOcclusionReal(p, n);
	float sha = softshadow2(p, lightDir, 0.01, length(lightPos - p), 4.0);
	//float sha = shadow(p, lightDir, 0.1, length(lightPos - p));
	float light = clamp(dot(n, lightDir), 0.0, 1.0); // lambert lighting
	float sky = clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
	float ind = clamp(dot(n, normalize(lightDir*vec3(-1.0,0.0,-1.0))), 0.0, 1.0); // indirect lighting
	//vec3 shading = light * vec3(1.64,1.27,0.99) * pow(vec3(sha),vec3(1.0,1.2,1.5));
	vec3 shading = phongContribForLight(
		vec3(1.64,1.27,0.99), // diffuse
		vec3(1.0, 1.0, 0.0), 1280.0, // specular
		p, ro, lightPos, vec3(1.0)) * pow(vec3(sha),vec3(1.0,1.2,1.5));
    shading += sky * vec3(0.16,0.20,0.28) * occ;
    shading += ind * vec3(0.40,0.28,0.20) * occ;
	
	// DOESN'T WORK!
	//float fresnel = clamp(1.0 + dot(n, rd), 0.0, 1.0);
	//shading += 25.0 * fresnel * fresnel * occ;
	
	c = c * shading;

	// add fog/haze
	//c = applyFog(c, ro, p, vec3(0.34, 0.435, 0.57));
	c = applyScattering(c, ro, p, vec3(0.34, 0.435, 0.57), vec3(2.0), vec3(2.0));

	// color post processing
	
	//c = c / (c + 1); // reinhard
	//vec3 x = max(vec3(0.0), c - 0.004); // filmic curve
	//c = (x * (6.2 * x + .5)) / (x*(6.2 * x + 1.7) + 0.06);
	
	//c = gammaCorrection(c);
	c = tonemap(c); // uses gamma correction already
	c = contrast(c);

    return c;
}

mat2 rot(float a) {
	float s = sin(a);
	float c = cos(a);
	return mat2(c, -s, s, c);
}

/**
 * Return the normalized direction to march in from the eye point for a single pixel.
 * 
 * fieldOfView: vertical field of view in degrees
 * size: resolution of the output image
 * fragCoord: the x,y coordinate of the pixel in the output image
 */
vec3 calcRayDir(float fieldOfView, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size / 2.0;
    float z = size.y / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

// NO AA
void main() {
	vec2 uv = (gl_TexCoord[0].xy - 0.5) * u_resolution / u_resolution.y;
	vec3 rayOrigin = u_pos;
	
	// default perspective projection
	//vec3 rayDirection = normalize(vec3(uv.x, -uv.y, -1.0));
	
	// fisheye
	float curve_coeff = radians(110.0f /*fov*/) * 0.5;
	vec3 rayDirection = vec3(0.0, 0.0, -1.0);
	rayDirection.yz *= rot(-uv.y * curve_coeff);
	rayDirection.xz *= rot(uv.x * curve_coeff);
	
	// rotate camera
	rayDirection.yz *= rot(-u_mouse.y);
	rayDirection.xz *= rot(u_mouse.x);
	
	// render image
	vec3 col = render(rayOrigin, rayDirection);
	col = vignette(col, gl_TexCoord[0].xy);
	//col = interlacedScan(col, gl_TexCoord[0].xy);
	gl_FragColor = vec4(col, 1.0);
}

// SSAA (2x)
//void main() {
//	vec3 sum_col = vec3(0.0);
//	int AA = 2;
//	for (int jj=0; jj<AA; jj++)
//    for (int ii=0; ii<AA; ii++)
//	{
//		// gl_TexCoord[0].xy range: [0, 1]
//		vec2 uv = (gl_TexCoord[0].xy - 0.5) * u_resolution / u_resolution.y;
//		vec2 aa_offset = vec2(float(ii), float(jj)) / (float(AA) * u_resolution.y);
//		uv += aa_offset;
//		
//		vec3 rayOrigin = u_pos;
//		vec3 rayDirection = normalize(vec3(uv.x, -uv.y, -1.0));
//		rayDirection.yz *= rot(-u_mouse.y);
//		rayDirection.xz *= rot(u_mouse.x);
//		sum_col += render(rayOrigin, rayDirection);
//	}
//	vec3 col = sum_col / float(AA*AA);
//	
//	col = vignette(col, gl_TexCoord[0].xy);
//	
//	gl_FragColor = vec4(col, 1.0);
//}
