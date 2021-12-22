#version 130

uniform vec2 u_resolution;
uniform sampler2D u_main_tex;

// - FXAA ---------------------------------------------------------------------------------

// based on:
// https://github.com/mattdesl/glsl-fxaa
// https://developer.download.nvidia.com/assets/gamedev/files/sdk/11/FXAA_WhitePaper.pdf

#define FXAA_REDUCE_MIN   (1.0/ 128.0)
#define FXAA_REDUCE_MUL   (1.0 / 8.0)
#define FXAA_SPAN_MAX     8.0

vec4 fxaa(sampler2D tex, vec2 fragCoord, vec2 resolution)
{
    // local contrast check (find edges)
	vec2 inverseVP = 1.0 / resolution;
    vec3 rgbNW = texture(tex, fragCoord + vec2(-1.0, -1.0) * inverseVP).xyz;
    vec3 rgbNE = texture(tex, fragCoord + vec2(1.0, -1.0) * inverseVP).xyz;
    vec3 rgbSW = texture(tex, fragCoord + vec2(-1.0, 1.0) * inverseVP).xyz;
    vec3 rgbSE = texture(tex, fragCoord + vec2(1.0, 1.0) * inverseVP).xyz;
    vec4 texColor = texture(tex, fragCoord);
    vec3 rgbM  = texColor.xyz;
	
    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM,  luma);
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    
	vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));
    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) *
                          (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
	float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
              max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
              dir * rcpDirMin)) * inverseVP;
    
	// blur edges
	vec3 rgbA = 0.5 * (
        texture(tex, fragCoord + dir * (1.0 / 3.0 - 0.5)).xyz +
        texture(tex, fragCoord + dir * (2.0 / 3.0 - 0.5)).xyz);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture(tex, fragCoord + dir * -0.5).xyz +
        texture(tex, fragCoord + dir * 0.5).xyz);
    
    vec4 color;
	float lumaB = dot(rgbB, luma);
    if ((lumaB < lumaMin) || (lumaB > lumaMax))
        color = vec4(rgbA, texColor.a);
    else
        color = vec4(rgbB, texColor.a);
    return color;
}

// - TONEMAPPING --------------------------------------------------------------------------
// useful link about tonemapping: https://64.github.io/tonemapping/

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 tonemapACES(vec3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

vec3 tonemapReinhard(vec3 c)
{
	return c / (c + 1.0);
}

vec3 tonemapReinhardJodie(vec3 c)
{
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722)); // luminance
    vec3 tc = c / (c + 1.0); // reinhard

    return mix(c / (l + 1.0), tc, tc);
}

vec3 tonemapUncharted2Partial(vec3 x)
{
    float A = 0.15f;
    float B = 0.50f;
    float C = 0.10f;
    float D = 0.20f;
    float E = 0.02f;
    float F = 0.30f;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

vec3 tonemapUncharted2Filmic(vec3 v)
{
    float exposure_bias = 2.0f;
    vec3 curr = tonemapUncharted2Partial(v * exposure_bias);

    vec3 W = vec3(11.2f);
    vec3 white_scale = vec3(1.0f) / tonemapUncharted2Partial(W);
    return curr * white_scale;
}

// - CHROMATIC ABERRATIONS ----------------------------------------------------------------

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

// ----------------------------------------------------------------------------------------

void main()
{
	vec4 color;
	
	// apply antialiasing (FXAA)
	vec2 uv = vec2(gl_TexCoord[0].x, 1.0 - gl_TexCoord[0].y);
    color = fxaa(u_main_tex, uv, u_resolution);
	
	gl_FragColor = color;
}