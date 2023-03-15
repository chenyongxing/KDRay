#include "scene_loader.h"
#include "scene.h"
#include "renderer.h"
#include <rapidxml/rapidxml.hpp>
#include <rapidxml/rapidxml_utils.hpp>

namespace
{
	inline bool SplitStringToFloat2(const std::string& str, float2* f2)
	{
		float vec[2];
		size_t i = 0;

		std::string s = str;
		std::string delimiter = " ";
		size_t pos = 0;
		std::string token;
		while ((pos = s.find(delimiter)) != std::string::npos)
		{
			token = s.substr(0, pos);
			s.erase(0, pos + delimiter.length());

			if (i < 3)
			{
				float number = (float)std::atof(token.c_str());
				vec[i] = number;
				i++;
			}
		}

		float number = (float)std::atof(s.c_str());
		vec[i] = number;
		i++;

		memcpy(f2, vec, sizeof(float) * 2);
		return i == 3;
	}

	inline bool SplitStringToFloat3(const std::string& str, float3* f3)
	{
		float vec[3];
		size_t i = 0;

		std::string s = str;
		std::string delimiter = " ";
		size_t pos = 0;
		std::string token;
		while ((pos = s.find(delimiter)) != std::string::npos)
		{
			token = s.substr(0, pos);
			s.erase(0, pos + delimiter.length());

			if (i < 3)
			{
				float number = (float)std::atof(token.c_str());
				vec[i] = number;
				i++;
			}
		}

		float number = (float)std::atof(s.c_str());
		vec[i] = number;
		i++;

		memcpy(f3, vec, sizeof(float) * 3);
		return i == 3;
	}

	inline bool SplitStringToTransform(const std::string& str, float transform[16])
	{
		size_t i = 0;

		std::string s = str;
		std::string delimiter = " ";
		size_t pos = 0;
		std::string token;
		while ((pos = s.find(delimiter)) != std::string::npos)
		{
			token = s.substr(0, pos);
			s.erase(0, pos + delimiter.length());

			if (i < 16)
			{
				float number = (float)std::atof(token.c_str());
				transform[i] = number;
				i++;
			}
		}

		float number = (float)std::atof(s.c_str());
		transform[i] = number;
		i++;

		std::swap(transform[1], transform[4]);
		std::swap(transform[2], transform[8]);
		std::swap(transform[3], transform[12]);
		std::swap(transform[6], transform[9]);
		std::swap(transform[7], transform[13]);
		std::swap(transform[11], transform[14]);
		return i == 16;
	}
}

namespace kdray
{
	SceneLoader::SceneLoader(Scene* scene) :_scene(scene)
	{
	}

	bool SceneLoader::LoadFromFile(const std::string& filepath)
	{
		if (filepath.find("xml") == std::string::npos)	return false;
		std::string sceneDir = filepath.substr(0, filepath.find_last_of("/") + 1);
		if (sceneDir.empty())
		{
			sceneDir = filepath.substr(0, filepath.find_last_of("\\") + 1);
		}

		using namespace rapidxml;
		rapidxml::file<> fdoc(filepath.c_str());
		if (!fdoc.data())	return false;

		std::unique_ptr<xml_document<>> doc = std::make_unique<xml_document<>>();
		doc->parse<0>(fdoc.data());

		xml_node<>* root = doc->first_node();
		if (!root)	return false;

		_scene->Clear();

		for (xml_node<>* child = root->first_node(); child != nullptr; child = child->next_sibling())
		{
			if (std::string(child->name()) == "setting")
			{
				RenderSetting setting;
				for (xml_attribute<>* attr = child->first_attribute();
					attr != nullptr; attr = attr->next_attribute())
				{
					if (std::string(attr->name()) == "spp")
					{
						setting.sampleCount = std::stoi(attr->value());
					}
					else if (std::string(attr->name()) == "bounce")
					{
						setting.maxBounce = std::stoi(attr->value());
					}
					else if (std::string(attr->name()) == "exposure")
					{
						setting.exposure = std::stof(attr->value());
					}
					else if (std::string(attr->name()) == "toneMap")
					{
						setting.toneMap = (ToneMapType)std::stoi(attr->value());
					}
				}
				Renderer::Instance()->SetRenderSetting(setting);
			}
			else if (std::string(child->name()) == "camera")
			{
				std::string type;
				int width = 800;
				int height = 600;
				for (xml_attribute<>* attr = child->first_attribute();
					attr != nullptr; attr = attr->next_attribute())
				{
					if (std::string(attr->name()) == "type")
					{
						type = attr->value();
					}
					else if (std::string(attr->name()) == "width")
					{
						width = std::atoi(attr->value());
					}
					else if (std::string(attr->name()) == "height")
					{
						height = std::atoi(attr->value());
					}
				}
				Framebuffer* framebuffer = Renderer::Instance()->GetFramebuffer();
				framebuffer->Resize(width, height);

				if (type == "perspective")
				{
					for (xml_node<>* _child = child->first_node();
						_child != nullptr; _child = _child->next_sibling())
					{
						if (std::string(_child->name()) == "property")
						{
							float fovY = 45.f;
							float nearZ = 0.1f;
							float farZ = 100.f;
							for (xml_attribute<>* attr = _child->first_attribute();
								attr != nullptr; attr = attr->next_attribute())
							{
								if (std::string(attr->name()) == "fovY")
								{
									fovY = (float)std::atof(attr->value());
									fovY = std::max(1.0f, fovY);
								}
								else if (std::string(attr->name()) == "nearZ")
								{
									nearZ = (float)std::atof(attr->value());
									nearZ = std::max(0.0001f, nearZ);
								}
								else if (std::string(attr->name()) == "farZ")
								{
									farZ = (float)std::atof(attr->value());
								}
							}
							Camera* camera = Renderer::Instance()->GetCamera();
							camera->Perspect(fovY, (float)width / (float)height, nearZ, farZ);
						}
						else if (std::string(_child->name()) == "transform")
						{
							for (xml_attribute<>* _attr = _child->first_attribute();
								_attr != nullptr; _attr = _attr->next_attribute())
							{
								if (std::string(_attr->name()) == "matrix")
								{
									float transform[16];
									if (SplitStringToTransform(_attr->value(), transform))
									{
										Renderer::Instance()->GetCamera()->SetTransformMartix(transform);
									}
								}
							}
						}
					}
				}
			}
			else if (std::string(child->name()) == "light")
			{
				for (xml_attribute<>* attr = child->first_attribute();
					attr != nullptr; attr = attr->next_attribute())
				{
					Light* light = new Light();
					if (std::string(attr->name()) == "type")
					{
						std::string type(attr->value());
						if (type == "directional")
						{
							light->type = LightType::Directional;
						}
						else if (type == "envmap")
						{
							light->type = LightType::Envmap;
						}
						else if (type == "point")
						{
							light->type = LightType::Point;
						}
						else if (type == "rect")
						{
							light->type = LightType::Rect;
						}
						else if (type == "disk")
						{
							light->type = LightType::Disk;
						}
						else if (type == "spot")
						{
							light->type = LightType::Spot;
						}
					}

					for (xml_node<>* _child = child->first_node();
						_child != nullptr; _child = _child->next_sibling())
					{
						if (std::string(_child->name()) == "property")
						{
							for (xml_attribute<>* attr = _child->first_attribute();
								attr != nullptr; attr = attr->next_attribute())
							{
								if (std::string(attr->name()) == "radiance")
								{
									SplitStringToFloat3(attr->value(), &light->radiance);
								}
								else if (std::string(attr->name()) == "radius")
								{
									light->radius = (float)std::atof(attr->value());
								}
								else if (std::string(attr->name()) == "width")
								{
									light->width = (float)std::atof(attr->value());
								}
								else if (std::string(attr->name()) == "height")
								{
									light->height = (float)std::atof(attr->value());
								}
								else if (std::string(attr->name()) == "angularDiameter")
								{
									light->angularDiameter = (float)std::atof(attr->value());
								}
								else if (std::string(attr->name()) == "outerConeAngle")
								{
									light->outerConeAngle = (float)std::atof(attr->value());
								}
								else if (std::string(attr->name()) == "innerConeAngle")
								{
									light->innerConeAngle = (float)std::atof(attr->value());
								}
								else if (std::string(attr->name()) == "visible")
								{
									if (std::string(attr->value()) == "true")
									{
										light->visible = true;
									}
								}
								else if (std::string(attr->name()) == "doubleSided")
								{
									if (std::string(attr->value()) == "true")
									{
										light->doubleSided = true;
									}
								}
								else if (std::string(attr->name()) == "texture")
								{
									Texture* texture = new Texture();
									if (texture->LoadFromFile(sceneDir + attr->value()))
									{
										_scene->AddTexture(texture);
										light->texture = texture;
									}
									else
									{
										delete texture;
									}
								}
							}
						}
						else if (std::string(_child->name()) == "transform")
						{
							for (xml_attribute<>* _attr = _child->first_attribute();
								_attr != nullptr; _attr = _attr->next_attribute())
							{
								if (std::string(_attr->name()) == "matrix")
								{
									float transform[16];
									if (SplitStringToTransform(_attr->value(), transform))
									{
										_scene->SetLightTransform(light, transform);
									}
								}
							}
						}
					}

					if (light->type != LightType::None)
					{
						_scene->AddLight(light);
					}
					else
					{
						delete light;
					}
				}
			}
			else if (std::string(child->name()) == "material")
			{
				Material* material = new Material();

				for (xml_attribute<>* attr = child->first_attribute();
					attr != nullptr; attr = attr->next_attribute())
				{
					if (std::string(attr->name()) == "type")
					{
						std::string type(attr->value());
						if (type == "metal")
						{
							material->type = MaterialType::Metal;
						}
						else if (type == "plastic")
						{
							material->type = MaterialType::Plastic;
						}
						else if (type == "glass")
						{
							material->type = MaterialType::Glass;
						}
						else if (type == "principled")
						{
							material->type = MaterialType::Principled;
						}
						else
						{
							material->type = MaterialType::Diffuse;
						}
					}
					else if (std::string(attr->name()) == "name")
					{
						material->name = attr->value();
					}
				}

				for (xml_node<>* _child = child->first_node();
					_child != nullptr; _child = _child->next_sibling())
				{
					if (std::string(_child->name()) == "property")
					{
						for (xml_attribute<>* attr = _child->first_attribute();
							attr != nullptr; attr = attr->next_attribute())
						{
							if (std::string(attr->name()) == "emission")
							{
								SplitStringToFloat3(attr->value(), &material->emission);
							}
							else if (std::string(attr->name()) == "opacity")
							{
								material->opacity = (float)std::atof(attr->value());
							}
							else if (std::string(attr->name()) == "color" || std::string(attr->name()) == "baseColor")
							{
								SplitStringToFloat3(attr->value(), &material->color);
							}
							else if (std::string(attr->name()) == "roughness")
							{
								material->roughness = (float)std::atof(attr->value());
							}
							else if (std::string(attr->name()) == "ior")
							{
								material->ior = (float)std::atof(attr->value());
							}
							else if (std::string(attr->name()) == "eta")
							{
								SplitStringToFloat3(attr->value(), &material->eta);
							}
							else if (std::string(attr->name()) == "k")
							{
								SplitStringToFloat3(attr->value(), &material->k);
							}
							else if (std::string(attr->name()) == "metallic")
							{
								material->metallic = (float)std::atof(attr->value());
							}
							else if (std::string(attr->name()) == "specular")
							{
								material->specular = (float)std::atof(attr->value());
							}
							else if (std::string(attr->name()) == "specularTint")
							{
								material->specularTint = (float)std::atof(attr->value());
							}
							else if (std::string(attr->name()) == "subsurface")
							{
								material->subsurface = (float)std::atof(attr->value());
							}
							else if (std::string(attr->name()) == "sheen")
							{
								material->sheen = (float)std::atof(attr->value());
							}
							else if (std::string(attr->name()) == "sheenTint")
							{
								material->sheenTint = (float)std::atof(attr->value());
							}
							else if (std::string(attr->name()) == "clearcoat")
							{
								material->clearcoat = (float)std::atof(attr->value());
							}
							else if (std::string(attr->name()) == "clearcoatGloss")
							{
								material->clearcoatGloss = (float)std::atof(attr->value());
							}
							else if (std::string(attr->name()) == "transmission")
							{
								material->transmission = (float)std::atof(attr->value());
							}
							else if (std::string(attr->name()) == "colorTexture")
							{
								Texture* texture = new Texture();
								texture->isSRGB = true;
								if (texture->LoadFromFile(sceneDir + attr->value()))
								{
									_scene->AddTexture(texture);
									material->colorTexture = texture;
								}
								else
								{
									delete texture;
								}
							}
							else if (std::string(attr->name()) == "colorTextureScale")
							{
								Texture* texture = material->colorTexture;
								if (texture)
								{
									SplitStringToFloat2(attr->value(), &texture->scale);
								}
							}
							else if (std::string(attr->name()) == "colorTextureOffset")
							{
								Texture* texture = material->colorTexture;
								if (texture)
								{
									SplitStringToFloat2(attr->value(), &texture->offset);
								}
							}
							else if (std::string(attr->name()) == "aoRoughMetalTexture")
							{
								Texture* texture = new Texture();
								if (texture->LoadFromFile(sceneDir + attr->value()))
								{
									_scene->AddTexture(texture);
									material->aoRoughMetalTexture = texture;
								}
								else
								{
									delete texture;
								}
							}
							else if (std::string(attr->name()) == "normalTexture")
							{
								Texture* texture = new Texture();
								if (texture->LoadFromFile(sceneDir + attr->value()))
								{
									_scene->AddTexture(texture);
									material->normalTexture = texture;
								}
								else
								{
									delete texture;
								}
							}
							else if (std::string(attr->name()) == "bumpTexture")
							{
								Texture* texture = new Texture();
								if (texture->LoadFromFile(sceneDir + attr->value()))
								{
									_scene->AddTexture(texture);
									material->bumpTexture = texture;
								}
								else
								{
									delete texture;
								}
							}
						}
					}
				}

				_scene->AddMaterial(material);
			}
			else if (std::string(child->name()) == "mesh")
			{
				std::string filepath;
				std::string material;

				for (xml_attribute<>* attr = child->first_attribute();
					attr != nullptr; attr = attr->next_attribute())
				{
					if (std::string(attr->name()) == "filepath" &&
						!std::string(attr->value()).empty())
					{
						filepath = attr->value();
					}
					else if (std::string(attr->name()) == "material")
					{
						material = attr->value();
					}
				}

				float transform[16];
				bool transformValid = false;
				for (xml_node<>* _child = child->first_node();
					_child != nullptr; _child = _child->next_sibling())
				{
					if (std::string(_child->name()) == "transform")
					{
						for (xml_attribute<>* _attr = _child->first_attribute();
							_attr != nullptr; _attr = _attr->next_attribute())
						{
							if (std::string(_attr->name()) == "matrix")
							{
								if (SplitStringToTransform(_attr->value(), transform))
								{
									transformValid = true;
								}
							}
						}
					}
				}

				Mesh* mesh = new Mesh();
				if (mesh->LoadFromFile(sceneDir + filepath))
				{
					_scene->AddMesh(mesh);
					Instance* instance = new Instance();
					instance->mesh = mesh;
					instance->material = _scene->GetMaterialByName(material);
					if (transformValid)
					{
						memcpy(instance->transform, transform, sizeof(float) * 16);
					}
					_scene->AddInstance(instance);
				}
			}
		}

		Renderer::Instance()->ResetAccumCount();
		return true;
	}
}