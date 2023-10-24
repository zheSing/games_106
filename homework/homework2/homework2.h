/*
* Vulkan Example - Variable rate shading
*
* Copyright (C) 2020 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"

#define ENABLE_VALIDATION false

class VulkanExample : public VulkanExampleBase
{
public:
	vkglTF::Model scene;

	struct ShadingRateImage {
		VkImage image;
		VkDeviceMemory memory;
		VkImageView view;
	} shadingRateImage;

	vks::Texture2D computeTarget;

	struct FrameBufferAttachment {
		VkImage image = VK_NULL_HANDLE;
		VkDeviceMemory mem = VK_NULL_HANDLE;
		VkImageView view = VK_NULL_HANDLE;
	};

	// compute shader
	std::vector<std::string> shaderNames;

	// Resources for the compute part of the example
	struct Compute {
		VkQueue queue;								// Separate queue for compute commands (queue family may differ from the one used for graphics)
		VkCommandPool commandPool;					// Use a separate command pool (queue family may differ from the one used for graphics)
		VkCommandBuffer commandBuffer;				// Command buffer storing the dispatch commands and barriers
		VkSemaphore semaphore;                      // Execution dependency between compute & graphic submission
		VkDescriptorSetLayout descriptorSetLayout;	// Compute shader binding layout
		VkDescriptorSet descriptorSet;				// Compute shader bindings
		VkPipelineLayout pipelineLayout;			// Layout of the compute pipeline
		std::vector<VkPipeline> pipelines;			// Compute pipelines for image filters
		int32_t pipelineIndex = 0;					// Current image filtering compute pipeline index
	} compute;

	struct PreDepthPass {
		uint32_t width, height;
		VkFramebuffer frameBuffer;
		FrameBufferAttachment depth;
		VkRenderPass renderPass;
		VkDescriptorImageInfo descriptor;
		VkPipelineLayout preDepthPipelineLayout;
		VkPipeline preDepthPipeline;
	} preDepthPass;


	bool enableShadingRate = true;
	bool colorShadingRate = false;

	struct ShaderData {
		vks::Buffer buffer;
		struct Values {
			glm::mat4 projection;
			glm::mat4 view;
			glm::mat4 model = glm::mat4(1.0f);
			glm::vec4 lightPos = glm::vec4(0.0f, 2.5f, 0.0f, 1.0f);
			glm::vec4 viewPos;
			int32_t colorShadingRate;
		} values;
	} shaderData;

	struct Pipelines {
		VkPipeline opaque;
		VkPipeline masked;
	};

	Pipelines basePipelines;
	Pipelines shadingRatePipelines;

	VkPipelineLayout pipelineLayout;
	VkDescriptorSet descriptorSet;
	VkDescriptorSetLayout descriptorSetLayout;

	VkPhysicalDeviceShadingRateImagePropertiesNV physicalDeviceShadingRateImagePropertiesNV{};
	VkPhysicalDeviceShadingRateImageFeaturesNV enabledPhysicalDeviceShadingRateImageFeaturesNV{};
	PFN_vkCmdBindShadingRateImageNV vkCmdBindShadingRateImageNV;

	VulkanExample();
	~VulkanExample();
	virtual void getEnabledFeatures();
	void handleResize();
	void buildCommandBuffers();
	void loadglTFFile(std::string filename);
	void loadAssets();
	void prepareShadingRateImage();
	void setupDescriptors();
	void preparePipelines();
	void prepareUniformBuffers();
	void updateUniformBuffers();
	void prepare();
	virtual void render();
	virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay);
	void preparePreDepthPass();
	void preparePreDepthImage();
	void createPreDepthPipeline();
	void createPreDepthFrameBuffer();
	void prepareComputeImage(uint32_t width, uint32_t height, VkFormat format);
	void prepareComputePipeline();
	void buildComputeCommandBuffer();
};