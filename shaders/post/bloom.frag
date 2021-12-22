#version 130

/*
Single pass bloom effect (medium quality)
=========================================
For correct effect you need to use:
	postTexture.setSmooth(true); // use bilinear filtering
	postTexture.generateMipmap(); // generate mip maps (because we use textureLod())
*/

uniform vec2 u_resolution;
uniform sampler2D u_main_tex;

mat3 gaussian_filter = mat3(41, 26, 7,
                           26, 16, 4,
                           7,  4,  1) / 273.;

vec3 bloom(float scale, float threshold, vec2 uv)
{
    float iaspect = u_resolution.y / u_resolution.x;
	float logScale = log2(scale * u_resolution.y);
    vec3 bloom = vec3(0);
    for(int y = -2; y <= 2; y++)
        for(int x = -2; x <= 2; x++)
			bloom += gaussian_filter[abs(x)][abs(y)] * textureLod(u_main_tex, (uv+vec2(x * iaspect, y)*scale), logScale).rgb;
    
    return max(bloom - vec3(threshold), vec3(0));
}

void main()
{
	// original image
	vec2 uv = vec2(gl_TexCoord[0].x, 1.0 - gl_TexCoord[0].y);
	vec3 color = texture(u_main_tex, uv).rgb;
	
	// apply bloom effect
	float bloom_scale = 0.05;
	float bloom_threshold = 0.3;
	float bloom_intensity = 1.0;
	color += bloom(bloom_scale, bloom_threshold, uv) * bloom_intensity;
	
	gl_FragColor = vec4(color, 1.0);
}