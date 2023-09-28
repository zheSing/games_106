#version 450

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;
layout (location = 5) in vec4 inTangent;

layout (location = 0) out vec4 outFragColor;

layout (set = 1, binding = 0) uniform sampler2D albedoMap;
layout (set = 1, binding = 1) uniform sampler2D metallicRoughnessMap;
layout (set = 1, binding = 2) uniform sampler2D normalMap;
layout (set = 1, binding = 3) uniform sampler2D emissiveMap;
layout (set = 1, binding = 4) uniform sampler2D occlusionMap;
//
//layout (set = 3, binding = 0) uniform Factors
//{
//	vec4 base;
//	vec3 emissive;
//	vec3 metallicRoughness;
//} factors;

//#define ALBEDO pow(texture(albedoMap, inUV).rgb, vec3(2.2))
#define ALBEDO texture(albedoMap, inUV).rgb

const float PI = 3.14159265359;

vec3 calculateNormal()
{
	vec3 tangentNormal = texture(normalMap, inUV).xyz * 2.0 - 1.0;

	vec3 N = normalize(inNormal);
	vec3 T = normalize(inTangent.xyz);
	vec3 B = normalize(cross(N, T));
	mat3 TBN = mat3(T, B, N);
	return normalize(TBN * tangentNormal);
}

// Normal Distribution function --------------------------------------
float D_GGX(float dotNH, float roughness)
{
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
	return (alpha2)/(PI * denom*denom); 
}

// Geometric Shadowing function --------------------------------------
float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r*r) / 8.0;
	float GL = dotNL / (dotNL * (1.0 - k) + k);
	float GV = dotNV / (dotNV * (1.0 - k) + k);
	return GL * GV;
}

// Fresnel function ----------------------------------------------------
vec3 F_Schlick(float cosTheta, float metallic, vec3 F0)
{
	vec3 F = F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0); 
	return F;    
}

vec3 BRDF(vec3 L, vec3 V, vec3 N, float metallic, float roughness, vec3 F0)
{
	// Precalculate vectors and dot products	
	vec3 H = normalize (V + L);
	float dotNV = clamp(dot(N, V), 0.0, 1.0);
	float dotNL = clamp(dot(N, L), 0.0, 1.0);
	float dotLH = clamp(dot(L, H), 0.0, 1.0);
	float dotNH = clamp(dot(N, H), 0.0, 1.0);

	// Light color fixed
	vec3 lightColor = vec3(1.0, 0.75, 0.75);

	vec3 color = vec3(0.0);

	if (dotNL > 0.0)
	{
		float rroughness = max(0.05, roughness);
		// D = Normal distribution (Distribution of the microfacets)
		float D = D_GGX(dotNH, roughness); 
		// G = Geometric shadowing term (Microfacets shadowing)
		float G = G_SchlicksmithGGX(dotNL, dotNV, rroughness);
		// F = Fresnel factor (Reflectance depending on angle of incidence)
		vec3 F = F_Schlick(dotNV, metallic, F0);

		vec3 spec = D * F * G / (4.0 * dotNL * dotNV);

		color += spec * dotNL * lightColor;
	}

	return color;
}

void main() 
{
	// vec4 color = texture(samplerColorMap, inUV) * vec4(inColor, 1.0);

	// vec3 N = normalize(inNormal);
	// vec3 L = normalize(inLightVec);
	// vec3 V = normalize(inViewVec);
	// vec3 R = reflect(L, N);
	// vec3 diffuse = max(dot(N, L), 0.15) * inColor;
	// vec3 specular = pow(max(dot(R, V), 0.0), 16.0) * vec3(0.75);
	// outFragColor = vec4(diffuse * color.rgb + specular, 1.0);		

	vec3 N = calculateNormal();
	vec3 V = normalize(inViewVec);
	vec3 L = normalize(inLightVec);
	vec3 R = reflect(L, N);

	float metallic = texture(metallicRoughnessMap, inUV).r;
	float roughness = texture(metallicRoughnessMap, inUV).g;
	
	vec3 F0 = vec3(0.04); 
	F0 = mix(F0, ALBEDO, metallic);

	vec3 color = BRDF(L, V, N, metallic, roughness, F0);

    vec3 ambient = ALBEDO * 0.3 * texture(occlusionMap, inUV).r;

    vec3 emission = texture(emissiveMap, inUV).rgb;

	outFragColor = vec4(color+emission+ambient, 1.0);
}