#version 130

uniform vec2 u_resolution;
uniform sampler2D u_main_tex;

mat3 gaussianFilter = mat3(41, 26, 7,
                           26, 16, 4,
                           7,  4,  1) / 273.;

vec3 bloom(float scale, float threshold, vec2 uv){
    float iaspect = u_resolution.y / u_resolution.x;
	float logScale = log2(scale);
    vec3 bloom = vec3(0);
    for(int y = -2; y <= 2; y++)
        for(int x = -2; x <= 2; x++)
			bloom += gaussianFilter[abs(x)][abs(y)] * textureLod(u_main_tex, (uv+vec2(x * iaspect, y)*scale), logScale).rgb;
			//bloom += gaussianFilter[abs(x)][abs(y)] * texture(u_main_tex, (uv+vec2(x * iaspect, y)*scale)).rgb;
    
    return max(bloom - vec3(threshold), vec3(0));
}

void main()
{
	vec2 uv = vec2(gl_TexCoord[0].x, 1.0 - gl_TexCoord[0].y);
	float bloom_scale = 0.02;
	float bloom_threshold = 0.0;
	float bloom_intensity = 1.0;
	gl_FragColor = vec4(bloom(bloom_scale, bloom_threshold, uv) * bloom_intensity, 1.0);
}