#include <map>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <spdlog/spdlog.h>
#include <rapidxml/rapidxml.hpp>
#include <rapidxml/rapidxml_utils.hpp>
#include <rapidxml/rapidxml_print.hpp>
#include "scene.h"

using namespace std;
using namespace rapidxml;
using namespace kdray;

inline std::string stringReplace(const std::string& _str, const std::string& s1, const std::string& s2)
{
	std::string str = _str;
	while (str.find(s1) != std::string::npos)
	{
		str.replace(str.find(s1), s1.length(), s2);
	}
	return str;
}

std::string ConvertTransformString(const std::string& str, bool isCamera = false)
{
	float transform[16];
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

	// 坐标系转换
	glm::mat4 matrix{ 1.0f };
	matrix[0][0] = transform[0];
	matrix[1][0] = transform[1];
	matrix[2][0] = transform[2];
	matrix[3][0] = transform[3];
	matrix[0][1] = transform[4];
	matrix[1][1] = transform[5];
	matrix[2][1] = transform[6];
	matrix[3][1] = transform[7];
	matrix[0][2] = transform[8];
	matrix[1][2] = transform[9];
	matrix[2][2] = transform[10];
	matrix[3][2] = transform[11];
	matrix[0][3] = transform[12];
	matrix[1][3] = transform[13];
	matrix[2][3] = transform[14];
	matrix[3][3] = transform[15];

	// x旋转90度
	glm::mat4 yUp2zUp
	{
		1.f, 0.f, 0.f, 0.f,
		0.f, 0.f, 1.f, 0.f,
		0.f, -1.f, 0.f, 0.f,
		0.f, 0.f, 0.f, 1.f,
	};
	matrix = yUp2zUp * matrix;

	if (isCamera)
	{
		// x=-x, z=-z
		glm::mat4 xform
		{
			-1.f, 0.f, 0.f, 0.f,
			0.f, 1.f, 0.f, 0.f,
			0.f, 0.f, -1.f, 0.f,
			0.f, 0.f, 0.f, 1.f,
		};
		matrix = matrix * xform;
	}
	
	transform[0] = matrix[0][0];
	transform[1] = matrix[1][0];
	transform[2] = matrix[2][0];
	transform[3] = matrix[3][0];
	transform[4] = matrix[0][1];
	transform[5] = matrix[1][1];
	transform[6] = matrix[2][1];
	transform[7] = matrix[3][1];
	transform[8] = matrix[0][2];
	transform[9] = matrix[1][2];
	transform[10] = matrix[2][2];
	transform[11] = matrix[3][2];
	transform[12] = matrix[0][3];
	transform[13] = matrix[1][3];
	transform[14] = matrix[2][3];
	transform[15] = matrix[3][3];

	std::string ostr;
	ostr.clear();
	ostr.append(std::to_string(transform[0]));
	ostr.append(" ");
	ostr.append(std::to_string(transform[1]));
	ostr.append(" ");
	ostr.append(std::to_string(transform[2]));
	ostr.append(" ");
	ostr.append(std::to_string(transform[3]));
	ostr.append(" ");
	ostr.append(std::to_string(transform[4]));
	ostr.append(" ");
	ostr.append(std::to_string(transform[5]));
	ostr.append(" ");
	ostr.append(std::to_string(transform[6]));
	ostr.append(" ");
	ostr.append(std::to_string(transform[7]));
	ostr.append(" ");
	ostr.append(std::to_string(transform[8]));
	ostr.append(" ");
	ostr.append(std::to_string(transform[9]));
	ostr.append(" ");
	ostr.append(std::to_string(transform[10]));
	ostr.append(" ");
	ostr.append(std::to_string(transform[11]));
	ostr.append(" ");
	ostr.append(std::to_string(transform[12]));
	ostr.append(" ");
	ostr.append(std::to_string(transform[13]));
	ostr.append(" ");
	ostr.append(std::to_string(transform[14]));
	ostr.append(" ");
	ostr.append(std::to_string(transform[15]));
	return ostr;
}

string find_child_attribute(xml_node<>* node, string attrName)
{
	xml_node<>* targetNode = nullptr;
	for (xml_node<>* child = node->first_node(); child != nullptr; child = child->next_sibling())
	{
		for (xml_attribute<>* attr = child->first_attribute(); attr != nullptr; attr = attr->next_attribute())
		{
			if (std::string(attr->name()) == "name" &&
				std::string(attr->value()) == attrName)
			{
				targetNode = child;
				break;
			}
		}
		if (targetNode)
		{
			break;
		}
	}
	if (targetNode)
	{
		for (xml_attribute<>* attr = targetNode->first_attribute(); attr != nullptr; attr = attr->next_attribute())
		{
			if (std::string(attr->name()) == "value")
			{
				return std::string(attr->value());
			}
		}
	}

	return "";
}

string get_transform_matrix(xml_node<>* node)
{
	for (xml_node<>* child = node->first_node(); child != nullptr; child = child->next_sibling())
	{
		if (std::string(child->name()) == "transform")
		{
			for (xml_node<>* _child = child->first_node(); _child != nullptr; _child = _child->next_sibling())
			{
				if (std::string(_child->name()) == "matrix")
				{
					for (xml_attribute<>* attr = _child->first_attribute(); attr != nullptr; attr = attr->next_attribute())
					{
						if (std::string(attr->name()) == "value")
						{
							return std::string(attr->value());
						}
					}
				}
			}
		}
	}
	return "";
}

string get_texture_filename(xml_node<>* node, string attrName)
{
	for (xml_node<>* child = node->first_node(); child != nullptr; child = child->next_sibling())
	{
		if (std::string(child->name()) == "texture")
		{
			for (xml_attribute<>* attr = child->first_attribute(); attr != nullptr; attr = attr->next_attribute())
			{
				if (std::string(attr->name()) == "name" && attr->value() == attrName)
				{
					return find_child_attribute(child, "filename");
				}
			}
		}
	}
	return "";
}

bool readScene(const std::string& filepath, Scene* scene)
{
	rapidxml::file<> fdoc(filepath.c_str());
	if (!fdoc.data())	return false;

	unique_ptr<xml_document<>> doc = make_unique<xml_document<>>();
	doc->parse<0>(fdoc.data());

	xml_node<>* root = doc->first_node();
	if (!root)	return false;

	std::map<std::string, std::string> defaultValues;
	for (xml_node<>* child = root->first_node(); child != nullptr; child = child->next_sibling())
	{
		if (std::string(child->name()) == "default")
		{
			std::string name;
			std::string value;
			for (xml_attribute<>* attr = child->first_attribute();
				attr != nullptr; attr = attr->next_attribute())
			{
				if (std::string(attr->name()) == "name")
				{
					name = attr->value();
				}
				else if (std::string(attr->name()) == "value")
				{
					value = attr->value();
				}
			}
			defaultValues[name] = value;
		}
		else if (std::string(child->name()) == "sensor")
		{
			scene->camera.type = "perspective";

			for (xml_node<>* _child = child->first_node();
				_child != nullptr; _child = _child->next_sibling())
			{
				if (std::string(_child->name()) == "film")
				{
					string width = find_child_attribute(_child, "width");
					scene->camera.width = width;
					string height = find_child_attribute(_child, "height");
					scene->camera.height = height;
				}
			}

			string fovStr = find_child_attribute(child, "fov");
			// 垂直fov转水平fov
			float width = std::stof(scene->camera.width);
			float height = std::stof(scene->camera.height);
			if (height > width)
			{
				float fov = std::stof(fovStr);
				float scale = height / width;
				fov = atanf(tanf(glm::radians(fov / 2.0f)) * scale) * 2.0f;
				fov = glm::degrees(fov);
				fovStr = std::to_string(fov);
			}
			scene->camera.fovY = fovStr;

			string transform = get_transform_matrix(child);
			if (!transform.empty())
			{
				scene->camera.transformMatrix = ConvertTransformString(transform, true);
			}
			else
			{
				spdlog::error("film transform error");
			}
		}
		else if (std::string(child->name()) == "bsdf")
		{
			string bsdfType;
			string id;
			for (xml_attribute<>* attr = child->first_attribute(); attr != nullptr; attr = attr->next_attribute())
			{
				if (std::string(attr->name()) == "type")
				{
					bsdfType = attr->value();
				}
				else if (std::string(attr->name()) == "id")
				{
					id = attr->value();
				}
			}

			auto processMaterial = [](xml_node<>* node)
			{
				Material material;
				xml_node<>* child = node->first_node();
				for (xml_attribute<>* attr = child->first_attribute(); attr != nullptr; attr = attr->next_attribute())
				{
					if (std::string(attr->name()) == "type")
					{
						string type = attr->value();
						if (type == "diffuse")
						{
							material.type = "diffuse";
							material.color = find_child_attribute(child, "reflectance");
							material.color = stringReplace(material.color, ",", "");
							material.colorTexture = get_texture_filename(child, "reflectance");
						}
						else if (type == "conductor")
						{
							material.type = "metal";
							material.color = find_child_attribute(child, "specularReflectance");
							material.color = stringReplace(material.color, ",", "");
							material.eta = find_child_attribute(child, "eta");
							material.eta = stringReplace(material.eta, ",", "");
							material.k = find_child_attribute(child, "k");
							material.k = stringReplace(material.k, ",", "");
							material.colorTexture = get_texture_filename(child, "specularReflectance");
						}
						else if (type == "roughconductor")
						{
							material.type = "metal";
							material.color = find_child_attribute(child, "specularReflectance");
							material.color = stringReplace(material.color, ",", "");
							material.roughness = find_child_attribute(child, "alpha");
							material.roughness = to_string(sqrtf(std::stof(material.roughness)));
							material.eta = find_child_attribute(child, "eta");
							material.eta = stringReplace(material.eta, ",", "");
							material.k = find_child_attribute(child, "k");
							material.k = stringReplace(material.k, ",", "");
							material.colorTexture = get_texture_filename(child, "specularReflectance");
						}
						else if (type == "roughplastic")
						{
							material.type = "plastic";
							material.color = find_child_attribute(child, "diffuseReflectance");
							material.color = stringReplace(material.color, ",", "");
							material.roughness = find_child_attribute(child, "alpha");
							material.roughness = to_string(sqrtf(std::stof(material.roughness)));
							material.ior = find_child_attribute(child, "intIOR");
							material.colorTexture = get_texture_filename(child, "diffuseReflectance");
						}
					}
				}
				return material;
			};

			if (bsdfType == "bumpmap")
			{
				for (xml_node<>* _child = child->first_node(); _child != nullptr; _child = _child->next_sibling())
				{
					if (std::string(_child->name()) == "bsdf")
					{
						for (xml_attribute<>* attr = _child->first_attribute(); 
							attr != nullptr; attr = attr->next_attribute())
						{
							if (std::string(attr->name()) == "id")
							{
								id = attr->value();
							}
						}

						Material material = processMaterial(_child);
						if (!material.type.empty())
						{
							material.name = id;
							material.bumpTexture = get_texture_filename(child, "map");
							scene->materials.emplace_back(material);
						}
						else
						{
							spdlog::error("bsdf error: {}", id);
						}
					}
				}
			}
			else if (bsdfType == "dielectric")
			{
				Material material;
				material.name = id;
				material.type = "glass";
				material.color = "1.0 1.0 1.0";
				material.roughness = "0.0";
				material.ior = find_child_attribute(child, "intIOR");
				scene->materials.emplace_back(material);
			}
			else
			{
				Material material = processMaterial(child);
				if (!material.type.empty())
				{
					material.name = id;
					scene->materials.emplace_back(material);
				}
				else
				{
					spdlog::error("bsdf error: {}", id);
				}
			}
		}
		else if (std::string(child->name()) == "shape")
		{
			bool valid = true;
			bool isLight = false;
			Mesh mesh;
			for (xml_attribute<>* attr = child->first_attribute();
				attr != nullptr; attr = attr->next_attribute())
			{
				if (std::string(attr->name()) == "type")
				{
					std::string type = attr->value();
					// 查找emitter是不是灯光
					for (xml_node<>* _child = child->first_node();
						_child != nullptr; _child = _child->next_sibling())
					{
						if (std::string(_child->name()) == "emitter")
						{
							Light light;
							if (type == "rectangle")
							{
								light.type = "rect";
								light.doubleSided = "true";
								light.width = "2.0";
								light.height = "2.0";
							}
							else if (type == "disk")
							{
								light.type = "disk";
								light.doubleSided = "true";
								light.radius = "0.1";
							}

							light.radiance = find_child_attribute(_child, "radiance");
							light.radiance = stringReplace(light.radiance, ",", "");

							string transform = get_transform_matrix(child);
							if (!transform.empty())
							{
								light.transformMatrix = ConvertTransformString(transform);
							}
							else
							{
								spdlog::error("light transform error");
							}
							scene->lights.emplace_back(light);
							isLight = true;
						}
					}
					if (isLight)	continue;

					if (type == "obj")
					{
						string filename = find_child_attribute(child, "filename");
						mesh.filepath = filename;
					}
					else if (type == "rectangle")
					{
						mesh.filepath = "mitsuba_rectangle.obj";
					}
					else if (type == "cube")
					{
						mesh.filepath = "mitsuba_cube.obj";
					}
					else
					{
						valid = false;
						spdlog::error("shape error type: {}", type);
					}
				}
			}
			if (isLight)	continue;

			string id;
			for (xml_node<>* _child = child->first_node();
				_child != nullptr; _child = _child->next_sibling())
			{
				if (std::string(_child->name()) == "ref")
				{
					for (xml_attribute<>* attr = _child->first_attribute();
						attr != nullptr; attr = attr->next_attribute())
					{
						if (std::string(attr->name()) == "id")
						{
							id = attr->value();
							mesh.material = id;
						}
					}
				}
			}
			if (id.empty())
			{
				valid = false;
				spdlog::error("shape error id");
			}

			if (valid)
			{
				string transform = get_transform_matrix(child);
				if (!transform.empty())
				{
					mesh.transformMatrix = ConvertTransformString(transform);
					scene->meshes.emplace_back(mesh);
				}
				else
				{
					spdlog::error("shape transform error");
				}
			}
		}
		else if (std::string(child->name()) == "emitter")
		{
			for (xml_attribute<>* attr = child->first_attribute();
				attr != nullptr; attr = attr->next_attribute())
			{
				if (std::string(attr->name()) == "type")
				{
					Light light;
					std::string type = attr->value();
					if (type == "envmap")
					{
						light.type = "envmap";
						light.texture = find_child_attribute(child, "filename");
					}

					string transform = get_transform_matrix(child);
					if (!transform.empty())
					{
						light.transformMatrix = ConvertTransformString(transform);
					}
					else
					{
						spdlog::error("light transform error");
					}
					scene->lights.emplace_back(light);
				}
			}
		}
	}

	return true;
}

void writeScene(const std::string& filepath, const Scene& scene)
{
	unique_ptr<xml_document<>> doc = make_unique<xml_document<>>();
	xml_node<>* root = doc->allocate_node(node_element, "scene");
	doc->append_node(root);

	xml_node<>* camera = doc->allocate_node(node_element, "camera");
	root->append_node(camera);
	camera->append_attribute(doc->allocate_attribute("type", scene.camera.type.c_str()));
	camera->append_attribute(doc->allocate_attribute("width", scene.camera.width.c_str()));
	camera->append_attribute(doc->allocate_attribute("height", scene.camera.height.c_str()));
	xml_node<>* camera_property = doc->allocate_node(node_element, "property");
	camera->append_node(camera_property);
	camera_property->append_attribute(doc->allocate_attribute("fovY", scene.camera.fovY.c_str()));
	xml_node<>* camera_transform = doc->allocate_node(node_element, "transform");
	camera->append_node(camera_transform);
	camera_transform->append_attribute(doc->allocate_attribute("matrix", scene.camera.transformMatrix.c_str()));
	
	for (auto& lightData : scene.lights)
	{
		xml_node<>* light = doc->allocate_node(node_element, "light");
		light->append_attribute(doc->allocate_attribute("type", lightData.type.c_str()));
		xml_node<>* light_property = doc->allocate_node(node_element, "property");
		light->append_node(light_property);
		if (!lightData.radiance.empty())
		{
			light_property->append_attribute(doc->allocate_attribute("radiance", lightData.radiance.c_str()));
		}
		if (lightData.type == "envmap")
		{
			light_property->append_attribute(doc->allocate_attribute("texture", lightData.texture.c_str()));
		}
		else if (lightData.type == "point")
		{
			light_property->append_attribute(doc->allocate_attribute("radius", lightData.radius.c_str()));
		}
		else if (lightData.type == "disk")
		{
			light_property->append_attribute(doc->allocate_attribute("doubleSided", lightData.doubleSided.c_str()));
			light_property->append_attribute(doc->allocate_attribute("radius", lightData.radius.c_str()));
		}
		else if (lightData.type == "rect")
		{
			light_property->append_attribute(doc->allocate_attribute("doubleSided", lightData.doubleSided.c_str()));
			light_property->append_attribute(doc->allocate_attribute("width", lightData.width.c_str()));
			light_property->append_attribute(doc->allocate_attribute("height", lightData.height.c_str()));
		}
		xml_node<>* light_transform = doc->allocate_node(node_element, "transform");
		light->append_node(light_transform);
		light_transform->append_attribute(doc->allocate_attribute("matrix", lightData.transformMatrix.c_str()));
		root->append_node(light);
	}

	for (auto& materialData : scene.materials)
	{
		xml_node<>* material = doc->allocate_node(node_element, "material");
		material->append_attribute(doc->allocate_attribute("name", materialData.name.c_str()));
		material->append_attribute(doc->allocate_attribute("type", materialData.type.c_str()));
		xml_node<>* material_property = doc->allocate_node(node_element, "property");
		material->append_node(material_property);
		if (!materialData.color.empty())
		{
			material_property->append_attribute(doc->allocate_attribute("color", materialData.color.c_str()));
		}
		if (!materialData.roughness.empty())
		{
			material_property->append_attribute(doc->allocate_attribute("roughness", materialData.roughness.c_str()));
		}
		if (!materialData.colorTexture.empty())
		{
			material_property->append_attribute(doc->allocate_attribute("colorTexture", materialData.colorTexture.c_str()));
		}
		if (!materialData.bumpTexture.empty())
		{
			material_property->append_attribute(doc->allocate_attribute("bumpTexture", materialData.bumpTexture.c_str()));
		}
		if (materialData.type == "plastic" || materialData.type == "glass")
		{
			material_property->append_attribute(doc->allocate_attribute("ior", materialData.ior.c_str()));
		}
		else if (materialData.type == "metal")
		{
			material_property->append_attribute(doc->allocate_attribute("eta", materialData.eta.c_str()));
			material_property->append_attribute(doc->allocate_attribute("k", materialData.k.c_str()));
		}
		root->append_node(material);
	}

	for (auto& meshData : scene.meshes)
	{
		xml_node<>* mesh = doc->allocate_node(node_element, "mesh");
		mesh->append_attribute(doc->allocate_attribute("filepath", meshData.filepath.c_str()));
		mesh->append_attribute(doc->allocate_attribute("material", meshData.material.c_str()));
		root->append_node(mesh);
		xml_node<>* mesh_transform = doc->allocate_node(node_element, "transform");
		mesh->append_node(mesh_transform);
		mesh_transform->append_attribute(doc->allocate_attribute("matrix", meshData.transformMatrix.c_str()));
	}
	
	std::string xml_as_string;
	rapidxml::print(std::back_inserter(xml_as_string), *doc.get());
	std::ofstream file_stored(filepath);
	file_stored << *doc.get();
	file_stored.close();
	doc->clear();
}

int main(int argc, char* argv[])
{
	string filepath = "material-testball/scene_v0.6.xml";
	if (argc > 2)
	{
		filepath = argv[1];
	}
	spdlog::info("open file {}", filepath);

	Scene scene;
	if (readScene(filepath, &scene))
	{
		std::string dir = filepath.substr(0, filepath.find_last_of("/") + 1);
		writeScene(dir + "scene.xml", scene);
	}

	return 0;
}
