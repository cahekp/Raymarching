#include "common.frag"

const Material red = Material(vec3(0.2, 0.02, 0.02), vec3(0.04, 0.02, 0.02), 32.0, 0.0);
const Material green = Material(vec3(0.02, 0.2, 0.02), vec3(0.02, 0.04, 0.02), 32.0, 0.0);
const Material blue = Material(vec3(0.02, 0.02, 0.2), vec3(0.02, 0.02, 0.04), 32.0, 0.0);
const Material mirror = Material(vec3(0.1), vec3(0.09), 64., 0.15);
Material floorMat(vec3 pos)
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

/**
 * Signed distance function describing the scene.
 * 
 * Absolute value of the return value indicates the distance to the surface.
 * Sign indicates whether the point is inside or outside the surface,
 * negative indicating inside.
 */
// distance field
SdResult sceneSDF(vec3 p)
{
	vec4 c = vec4(1);
	//SdResult dist0 = SdResult(mandelbulb(transformRS1(p - vec3(0, 2, 0), vec3(180, u_time * 2, 0), 1.5), c) * 1.5, green);
	SdResult dist0 = SdResult(mengersponge(transformR(p - vec3(0, 3, 0), vec3(180, u_time * 2, 0))).x, mirror);
	
	SdResult dist1 = SdResult(sphere(vec4(0, 0, 2, 1), p), red);
	SdResult dist2 = SdResult(cube(vec4(0, 0, 0, 1), p), blue);
	SdResult dist3 = SdResult(plane(p), floorMat(p));
	return sminCubic(dist0, sminCubic(sminCubic(dist1, dist2, 0.5), dist3, 0.5), 0.33);
}

vec3 light(Material mat, vec3 ro, vec3 rd, vec3 p, vec3 n)
{
	vec3 lightPos = vec3(20, 50, 0);
	vec3 lightDir = normalize(lightPos - p);

	float occ = ambientOcclusionReal(p, n);
	float sha = softshadow2(p, lightDir, 0.01, length(lightPos - p), 4.0);
	
	//float light = clamp(dot(n, lightDir), 0.0, 1.0); // lambert lighting
	
	float sky = clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
	float ind = clamp(dot(n, normalize(lightDir * vec3(-1.0,0.0,-1.0))), 0.0, 1.0); // indirect lighting
	float fre = pow(clamp(1.0 + dot(n, rd), 0.0, 1.0), 2.0); // fresnel effect
	// (dielectrics reflect from 0.05 to 30 percent of the light, while metals reflect from 55 up to 95 percent)
	
	//vec3 shading = light * vec3(1.64,1.27,0.99) * pow(vec3(sha),vec3(1.0,1.2,1.5));
	
	vec3 shading = phongContribForLight(
		//mat.diffuse, // diffuse
		vec3(1.64,1.27,0.99), // diffuse
		mat.specular, mat.shininess, // specular
		p, ro, lightPos, vec3(1.0)) * pow(vec3(sha),vec3(1.0,1.2,1.5));
	shading += sky * vec3(0.16,0.20,0.28) * occ;
	shading += ind * vec3(0.40,0.28,0.20) * occ;
	shading += fre * vec3(1.0,1.0,1.0) * occ;
	
	vec3 color = mat.diffuse * shading;

	// add fog/haze
	//color = applyFog(color, ro, p, vec3(0.34, 0.435, 0.57));
	color = applyScattering(color, ro, p, vec3(0.34, 0.435, 0.57), vec3(2.0), vec3(2.0));
	
	return color;
}

// TODO: move render() to common.frag
const float REFLECTION_EPS = 0.001;
const int MAX_REFLECTIONS = 3; // Maximum number of reflections. Total number of casts = 1+MAX_REFLECTIONS

// ro - ray origin
// rd - ray direction
// return: color of the intersection
vec3 render(in vec3 ro, in vec3 rd)
{
	vec3 color = vec3(0);
    float transmittance = 1.0;
	
	for (int i = 0; i <= MAX_REFLECTIONS; i++)
	{
		// find intersection point and its normal
		SdResult sd = castRayD(ro, rd);
		if (sd.dist > 0.0)
		{
			vec3 p = ro + rd * sd.dist;
			vec3 n = getNormalFast(p);
			
			color += transmittance * light(sd.mat, ro, rd, p, n);
			
			transmittance *= pow(sd.mat.reflectivity, 2.0);
			if (transmittance < REFLECTION_EPS)
				break;

			// prepare for the next reflection iteration
			ro = p + n * 0.001;
			rd = reflect(rd, n);
		}
		else
		{
			// skybox
			vec3 skyLight = vec3(0.4, 0.4, 0.8);
            vec3 skyDark = vec3(0.1, 0.1, 0.4);
            vec3 skyColor = mix(skyDark, skyLight, rd.y);
            //color += transmittance * skyColor;
			color = applyScattering(transmittance * color, ro, ro + rd * ZFAR, vec3(0.34, 0.435, 0.57), vec3(2.0), vec3(2.0));
			break;
		}
	}
	
    return color;
}

// NO AA
void main()
{
	vec2 uv = (gl_TexCoord[0].xy - 0.5) * u_resolution / u_resolution.y;
	vec3 rayOrigin = u_pos;
	
	// default perspective projection
	vec3 rayDirection = normalize(vec3(uv.x, -uv.y, -1.0));
	
	// fisheye
	//float curve_coeff = radians(110.0f /*fov*/) * 0.5;
	//vec3 rayDirection = vec3(0.0, 0.0, -1.0);
	//rayDirection.yz *= rot(-uv.y * curve_coeff);
	//rayDirection.xz *= rot(uv.x * curve_coeff);
	
	// rotate camera
	rayDirection.yz *= rot(-u_mouse.y);
	rayDirection.xz *= rot(u_mouse.x);
	
	// render image
	vec3 col = render(rayOrigin, rayDirection);
	
	// post processing
	//col = col / (col + 1); // reinhard
	//vec3 x = max(vec3(0.0), col - 0.004); // filmic curve
	//col = (x * (6.2 * x + .5)) / (x*(6.2 * x + 1.7) + 0.06);
	//col = gammaCorrection(col);
	col = tonemap(col); // uses gamma correction already
	col = contrast(col);
	col = vignette(col, gl_TexCoord[0].xy);
	//col = interlacedScan(col, gl_TexCoord[0].xy);
	
	gl_FragColor = vec4(col, 1.0);
}
