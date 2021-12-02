#include <common.frag>

// define material structure
struct Material
{
    vec3 diffuse;
    vec3 specular;
    float shininess;
    float reflectivity;
};

Material blendMaterial(Material a, Material b, float k)
{
    return Material(
        mix(a.diffuse, b.diffuse, k),
        mix(a.specular, b.specular, k),
        mix(a.shininess, b.shininess, k),
        mix(a.reflectivity, b.reflectivity, k)
    );
}

// all materials
const Material red = Material(vec3(0.2, 0.02, 0.02), vec3(0.04, 0.02, 0.02), 32.0, 0.0);
const Material mirror = Material(vec3(0.01), vec3(0.09), 64., 0.9);
Material floorMat(in vec3 pos)
{
    vec3 white = vec3(0.3);
    vec3 black = vec3(0.025);
    
    float smoothstepSize = 0.005;
    float scale = max(10., pow(length(pos), 1.3));
    vec2 tile2D = smoothstep(-smoothstepSize, smoothstepSize, sin(pos.xz * PI) / scale);
    float tile = min(max(tile2D.x, tile2D.y), max(1.-tile2D.x,1.-tile2D.y)); // Fuzzy xor.
    vec3 color = mix(white, black, tile);
    
    return Material(color,vec3(0.03), 128.0, 0.0);
}

SdResult sceneSDF(in vec3 p)
{
	float dist = mengersponge(transformR(p - vec3(0, 3, 0), vec3(180, u_time * 2, 0))).x;
	return dist;
}

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
	float light = clamp(dot(n, lightDir), 0.0, 1.0); // lambert lighting
	float sky = clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
	float ind = clamp(dot(n, normalize(lightDir*vec3(-1.0,0.0,-1.0))), 0.0, 1.0); // indirect lighting
	float fre = pow(clamp(1.0 + dot(n, rd), 0.0, 1.0), 2.0); // fresnel effect
	vec3 shading = phongContribForLight(
		vec3(1.64,1.27,0.99), // diffuse
		vec3(1.0, 1.0, 0.0), 1280.0, // specular
		p, ro, lightPos, vec3(1.0)) * pow(vec3(sha),vec3(1.0,1.2,1.5));
    shading += sky * vec3(0.16,0.20,0.28) * occ;
    shading += ind * vec3(0.40,0.28,0.20) * occ;
	shading += fre * vec3(1.0,1.0,1.0) * occ;
	c = c * shading;

	// add fog/haze
	c = applyScattering(c, ro, p, vec3(0.34, 0.435, 0.57), vec3(2.0), vec3(2.0));

    return c;
}

void main()
{
	vec2 uv = (gl_TexCoord[0].xy - 0.5) * u_resolution / u_resolution.y;
	vec3 rayOrigin = u_pos;
	
	// default perspective projection
	vec3 rayDirection = normalize(vec3(uv.x, -uv.y, -1.0));
	
	// rotate camera
	rayDirection.yz *= rot(-u_mouse.y);
	rayDirection.xz *= rot(u_mouse.x);
	
	// render image
	vec3 col = render(rayOrigin, rayDirection);
	
	// post processing
	col = tonemap(col); // uses gamma correction already
	col = contrast(col);
	col = vignette(col, gl_TexCoord[0].xy);

	gl_FragColor = vec4(col, 1.0);
}