#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>

class Node
{
public:
	virtual ~Node();

	virtual void Setup();
	virtual void Update(float deltaTime);

	void AddChild(Node* child);
	void RemoveChild(Node* child);
	inline void RemoveChild(int index) { children.erase(children.begin() + index); }

	void SetLocalTransform(glm::mat4& matrix);
	void SetLocalTransform(glm::mat4&& matrix);

	//世界变换，使用时候注意平移要在旋转缩放之后
	void SetPositon(glm::vec3& position);
	void SetPositon(glm::vec3&& position);
	void SetScale(glm::vec3& scale);
	void SetScale(glm::vec3&& scale);
	void SetRotation(glm::vec3& eulerAngles);
	void SetRotation(glm::vec3&& eulerAngles);
	void SetWorldTransform(glm::mat4& matrix);
	void SetWorldTransform(glm::mat4&& matrix);

	inline Node* GetChild(unsigned index)
	{
		return children[index];
	}

	inline unsigned GetChildrenCount()
	{
		return (unsigned)children.size();
	}

	inline glm::vec3 GetPositon()
	{
		return glm::vec3(localToWorldMatrix[3][0], localToWorldMatrix[3][1], localToWorldMatrix[3][2]);
	}

	inline const glm::mat4& LocalTransform()
	{
		return localTransform;
	}
	
	inline const glm::mat4& WorldTransform()
	{
		return localToWorldMatrix;
	}

	inline const glm::mat4& LocalToWorldMatrix()
	{
		return localToWorldMatrix;
	}

	inline const glm::mat4& WorldToLocalMatrix()
	{
		return worldToLocalMatrix;
	}

public:
	std::string name;
	bool visible = true;
	Node* parent = nullptr;
	
protected:
	std::vector<Node*> children;

	glm::mat4 localTransform{ 1.0f };
	glm::mat4 localToWorldMatrix{ 1.0f }; //worldTransform
	glm::mat4 worldToLocalMatrix{ 1.0f };

	bool localTransformDirty = false;
	bool worldTransformDirty = false;

	//自己变换发生变化时候，子节点需要保存本地变换不变更新世界变换
	void MarkWorldTransformDirty();
	void MarkLocalTransformDirty();
	void UpdateWorldTransformFromLoacl();
	void UpdateLocalTransformFromWorld();

	inline void UpdateWorldToLocalMatrix()
	{
		worldToLocalMatrix = glm::inverse(localToWorldMatrix);
	}
};

struct SubMesh
{
	unsigned int meshIndex = 0;
	unsigned int materialIndex = 0;
	unsigned int instanceIndex = 0;
	glm::vec3 boundboxMin{ 0.0f };
	glm::vec3 boundboxMax{ 0.0f };
};

class Mesh : public Node
{
public:
	std::vector<SubMesh> submeshes;
};
