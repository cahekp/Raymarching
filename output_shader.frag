#version 130
#define USE_MATERIAL_ID				// int id;
#define USE_MATERIAL_DIFFUSE		// vec3 diffuse;
#define USE_MATERIAL_SPECULAR		// vec3 specular;
#define USE_MATERIAL_SHININESS		// float shininess;
#define USE_MATERIAL_REFLECTIVITY	// float reflectivity;
#define USE_MATERIAL_REFRACTIVITY	// float refractivity;
#define ALLOW_MATERIAL_BLENDING		// smin() affects the material also
#include "common.frag"

const Material red = Material(vec3(0.2, 0.02, 0.02), vec3(0.04, 0.02, 0.02), 32.0, 0.0, 0.0);
const Material green = Material(vec3(0.02, 0.2, 0.02), vec3(0.02, 0.04, 0.02), 32.0, 0.0, 0.0);
const Material blue = Material(vec3(0.02, 0.02, 0.2), vec3(0.02, 0.02, 0.04), 32.0, 0.0, 1.0);
const Material mirror = Material(vec3(0.1), vec3(0.09), 64., 0.15, 0.0);
Material floorMat(vec3 pos)
{
    vec3 white = vec3(0.3);
    vec3 black = vec3(0.025);
    
    float smoothstepSize = 0.005;
    float scale = max(10., pow(length(pos), 1.3));
    vec2 tile2D = smoothstep(-smoothstepSize, smoothstepSize, sin(pos.xz * PI) / scale);
    float tile = min(max(tile2D.x, tile2D.y), max(1.-tile2D.x,1.-tile2D.y)); // Fuzzy xor.
    vec3 color = mix(white, black, tile);
    
    return Material(color,vec3(0.03), 128.0, 0.0, 0.0);
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
	
	SdResult dist1 = SdResult(sphere(vec4(3, 3, 3, 1), p), blue);
	SdResult dist2 = SdResult(cube(vec4(0, 1, 0, 1), p), blue);
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
	//float fre = pow(clamp(1.0 + dot(n, rd), 0.0, 1.0), 2.0); // fresnel effect
	// (dielectrics reflect from 0.05 to 30 percent of the light, while metals reflect from 55 up to 95 percent)
	
	//vec3 shading = light * vec3(1.64,1.27,0.99) * pow(vec3(sha),vec3(1.0,1.2,1.5));
	
	vec3 shading = phongContribForLight(
		//mat.diffuse, // diffuse
		vec3(1.64,1.27,0.99), // diffuse
		mat.specular, mat.shininess, // specular
		p, ro, lightPos, vec3(1.0)) * pow(vec3(sha),vec3(1.0,1.2,1.5));
	shading += sky * vec3(0.16,0.20,0.28) * occ;
	shading += ind * vec3(0.40,0.28,0.20) * occ;
	//shading += fre * vec3(1.0,1.0,1.0) * occ;
	
	vec3 color = mat.diffuse * shading;

	// add fog/haze
	//color = applyFog(color, ro, p, vec3(0.34, 0.435, 0.57));
	color = applyScattering(color, ro, p, vec3(0.34, 0.435, 0.57), vec3(2.0), vec3(2.0));
	
	return color;
}

vec3 background(vec3 ro, vec3 rd)
{
	vec3 color = vec3(0);
	return applyScattering(color, ro, ro + rd * ZFAR, vec3(0.34, 0.435, 0.57), vec3(2.0), vec3(2.0));
}

// refractive index of common materials:
// https://en.wikipedia.org/wiki/List_of_refractive_indices
#define REFRACTIVE_INDEX_OUTSIDE 1.00029 // air
#define REFRACTIVE_INDEX_INSIDE  1.5 //1.333 // water
#define OBJECT_DIFFUSE      vec3(0.0)
#define OBJECT_SPECPOWER    0.0
#define OBJECT_REFLECTIVITY 0.01 // How reflective the object is. regardless of fresnel.
#define OBJECT_ABSORB       vec3(2.0, 2.0, 0.75) // for beers law

// http://en.wikipedia.org/wiki/Schlick's_approximation
float Schlick( const in vec3 vHalf, const in vec3 vView, const in float fR0, const in float fSmoothFactor)
{
    float fDot = dot(vHalf, -vView);
    fDot = clamp((1.0 - fDot), 0.0, 1.0);
    float fDotPow = pow(fDot, 5.0);
    return fR0 + (1.0 - fR0) * fDotPow * fSmoothFactor;
}

vec3 ApplyFresnel(const in vec3 vDiffuse, const in vec3 vSpecular, const in vec3 vNormal, const in vec3 vView, in float fR0, float fSmoothness)
{
	vec3 vReflect = reflect(vView, vNormal);
	vec3 vHalf = normalize(vReflect + -vView);
    float fFresnel = Schlick(vHalf, vView, fR0, fSmoothness * 0.9 + 0.1);
    return mix(vDiffuse, vSpecular, fFresnel);
}

float fresnelReflectAmount(float n1, float n2, vec3 normal, vec3 incident)
{
	// schlick aproximation
	float r0 = (n1 - n2) / (n1 + n2);
	r0 *= r0;
	float cosX = -dot(normal, incident);
	if (n1 > n2)
	{
		float n = n1 / n2;
		float sinT2 = n * n * (1.0 - cosX * cosX);
		// total internal reflection
		if (sinT2 > 1.0)
			return 1.0;
		cosX = sqrt(1.0 - sinT2);
	}
	float x = 1.0 - cosX;
	float ret = r0 + (1.0 - r0) * x * x * x * x * x;

	// adjust reflect multiplier for object reflectivity
	ret = (1.0 - OBJECT_REFLECTIVITY) * ret + OBJECT_REFLECTIVITY;
	return ret;
}

#define MAX_REFLECTIONS 1
vec3 renderReflection(in vec3 ro, in vec3 rd, float reflectivity)
{
	vec3 color = vec3(0);
    float transmittance = 1.0;
	for (int i = 0; i < MAX_REFLECTIONS; i++)
	{
		SdResult sd = castRayD(ro, rd); // TODO: change to simple version
		if (sd.dist > 0.0) // render scene
		{
			// p - ray-surface intersection point
			// n - normal of the surface
			vec3 p = ro + rd * sd.dist;
			vec3 n = getNormalFast(p);
			
			color += transmittance * light(sd.mat, ro, rd, p, n);
			transmittance *= pow(reflectivity, 2.0);
			if (transmittance < 0.001 /* reflection epsilon */)
				break;

			// prepare for the next reflection iteration
			ro = p + n * 0.001;
			rd = reflect(rd, n);
		}
		else // render background (for example: skybox or gradient)
		{
			color += transmittance * background(ro, rd);
			break;
		}
	}
    return color;	
	
	// SINGLE REFLECTION
	//SdResult sd = castRayD(ro, rd);
	//if (sd.dist > 0.0) // render scene
	//{
	//	// p - ray-surface intersection point
	//	// n - normal of the surface
	//	vec3 p = ro + rd * sd.dist;
	//	vec3 n = getNormalFast(p);
	//	
	//	// light the surface
	//	vec3 color = light(sd.mat, ro, rd, p, n);
    //    return color;
	//}
	//else // render background (for example: skybox or gradient)
	//{
	//	return background(ro, rd);
	//}	
}

vec3 renderRefraction(in vec3 ro, in vec3 rd)
{
	// first pass: trace until outside the transparent object
	SdResult sd = castRayDI(ro, rd);
	if (sd.dist < 0.0) // when does it happen? when the object's size bigger than ZFAR
		return OBJECT_ABSORB;

	// p - ray-surface intersection point
	// n - normal of the surface
	vec3 p = ro + rd * sd.dist;
	vec3 n = -getNormalFast(p);
	
	// beer's law absorption
	vec3 absorb = exp(-OBJECT_ABSORB * sd.dist);
	
	// second pass: trace rest of the scene
	ro = p + rd * 0.001;
	rd = refract(rd, n, REFRACTIVE_INDEX_INSIDE);
	sd = castRayD(ro, rd);
	if (sd.dist > 0.0) // render scene
	{
		// p - ray-surface intersection point
		// n - normal of the surface
		vec3 p = ro + rd * sd.dist;
		vec3 n = getNormalFast(p);
		
		// light the surface
		return light(sd.mat, ro, rd, p, n) * absorb;
	}
	else // render background
	{
		return background(ro, rd) * absorb;
	}
	
	// ------------------------------------------------------------------------------
	

//    // bounce around inside the object as many times as needed (or until max bounces) due total internal reflection
//    float multiplier = 1.0;
//    vec3 ret = vec3(0.0);
//    float absorb_dist = 0.0;
//	for (int i = 0; i < MAX_RAY_BOUNCES; ++i)
//    {
//        // find intersection inside the object
//		SdResult sd = castRayDI(ro, rd);
//
//		// p - ray-surface intersection point (inside the object)
//		// n - normal of the surface
//		vec3 p = ro + rd * sd.dist;
//		vec3 n = getNormalFast(p);
//        
//        // calculate beer's law absorption
//        absorb_dist += sd.dist;
//        vec3 absorb = exp(-OBJECT_ABSORB * absorb_dist);
//        
//        // calculate how much to reflect or transmit (refract or diffuse)
//        float reflect_factor = fresnelReflectAmount(REFRACTIVE_INDEX_INSIDE, REFRACTIVE_INDEX_OUTSIDE, rd, n);
//        float refract_factor = 1.0 - reflect_factor;
//        
//        // add in refraction outside of the object
//        vec3 refracted_rd = refract(rd, n, REFRACTIVE_INDEX_INSIDE / REFRACTIVE_INDEX_OUTSIDE);
//        ret += renderReflection(p + refracted_rd * 0.001, refracted_rd) * refract_factor * multiplier * absorb;
//        
//        // add specular highlight based on refracted ray direction
//        //ret += light(mat, ro, rd, p, refracted_rd) * refract_factor * multiplier * absorb; 
//        
//        // follow the ray down the internal reflection path.
//        rd = reflect(rd, n);
//        
//        // move the ray slightly down the reflect path
//        ro = p + rd * 0.001;
//        
//        // For reflection, we are only going to be reflecting what is refracted on further bounces.
//        // So, we just need to make sure the next bounce is added in at the reflect_factor amount, recursively.
//		multiplier *= reflect_factor;        
//    }
//    
//    // return the color we calculated
//    return ret;	
}

// ro - ray origin
// rd - ray direction
// return: color of the intersection
vec3 render(in vec3 ro, in vec3 rd)
{
	SdResult sd = castRayD(ro, rd);
	if (sd.dist > 0.0) // render scene
	{
		// p - ray-surface intersection point
		// n - normal of the surface
		vec3 p = ro + rd * sd.dist;
		vec3 n = getNormalFast(p);
		
		// light the surface
		vec3 color = light(sd.mat, ro, rd, p, n);
		
        // calculate balance between reflection and transmission (diffuse or refract)
        float reflect_factor = fresnelReflectAmount(REFRACTIVE_INDEX_OUTSIDE, REFRACTIVE_INDEX_INSIDE, rd, n);
        float refract_factor = 1.0 - reflect_factor;
        
        // get reflection color
		if (sd.mat.reflectivity > 0)
		{
			vec3 reflected_rd = reflect(rd, n);
			color += renderReflection(p + reflected_rd * 0.001, reflected_rd, sd.mat.reflectivity) * reflect_factor;
		}
        
        // get refraction color
		if (sd.mat.transparency > 0)
		{
			vec3 refracted_rd = refract(rd, n, 1.0 / REFRACTIVE_INDEX_INSIDE);
			color += renderRefraction(p + refracted_rd * 0.001, refracted_rd);// * refract_factor;
		}
        
        return color;
	}
	else // render background (for example: skybox or gradient)
	{
		return background(ro, rd);
	}
}

// TODO: move render() to common.frag
//const float REFLECTION_EPS = 0.001;
//const int MAX_REFLECTIONS = 3; // Maximum number of reflections. Total number of casts = 1+MAX_REFLECTIONS
//
//// ro - ray origin
//// rd - ray direction
//// return: color of the intersection
//vec3 render(in vec3 ro, in vec3 rd)
//{
//	vec3 color = vec3(0);
//    float transmittance = 1.0;
//	
//	for (int i = 0; i <= MAX_REFLECTIONS; i++)
//	{
//		// find intersection point and its normal
//		SdResult sd = castRayD(ro, rd);
//		if (sd.dist > 0.0)
//		{
//			vec3 p = ro + rd * sd.dist;
//			vec3 n = getNormalFast(p);
//			
//			color += transmittance * light(sd.mat, ro, rd, p, n);
//			transmittance *= pow(sd.mat.reflectivity, 2.0);
//			if (transmittance < REFLECTION_EPS)
//				break;
//
//			// prepare for the next reflection iteration
//			ro = p + n * 0.001;
//			rd = reflect(rd, n);
//		}
//		else
//		{
//			// skybox
//			vec3 skyLight = vec3(0.4, 0.4, 0.8);
//            vec3 skyDark = vec3(0.1, 0.1, 0.4);
//            vec3 skyColor = mix(skyDark, skyLight, rd.y);
//			color += transmittance * applyScattering(color, ro, ro + rd * ZFAR, vec3(0.34, 0.435, 0.57), vec3(2.0), vec3(2.0));
//			break;
//		}
//	}
//	
//    return color;
//}

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
