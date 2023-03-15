#include "node.h"
#include <stdexcept>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

Node::~Node()
{
	for (Node* node : children)
	{
		if (node)
		{
			delete node;
		}
	}
}

void Node::Setup()
{
	for (Node* node : children)
	{
		node->Setup();
	}
}

void Node::Update(float deltaTime)
{
	if (!localTransformDirty && worldTransformDirty)
	{
		UpdateWorldToLocalMatrix();
		UpdateLocalTransformFromWorld();
	}
	else if (localTransformDirty && !worldTransformDirty)
	{
		UpdateWorldTransformFromLoacl();
		UpdateWorldToLocalMatrix();
	}
	else if(localTransformDirty && worldTransformDirty)
	{
		//用法错误，只能互相更新，一帧内同时被设置不支持
		throw std::runtime_error("");
	}

	for (Node* node : children)
	{
		node->Update(deltaTime);
	}
}

void Node::AddChild(Node* child)
{
	if (child == nullptr)
	{
		return;
	}
	child->parent = this;
	children.push_back(child);
}

void Node::RemoveChild(Node* child)
{
	for (auto it = children.begin(); it != children.end();)
	{
		if (*it == child) 
		{
			it = children.erase(it);
		}
		else 
		{
			it++;
		}
	}
}

void Node::SetLocalTransform(glm::mat4& matrix)
{
	localTransform = matrix;
	MarkLocalTransformDirty();
}

void Node::SetLocalTransform(glm::mat4&& matrix)
{
	SetLocalTransform(matrix);
}

void Node::SetPositon(glm::vec3& position)
{
	localToWorldMatrix[3][0] = position.x;
	localToWorldMatrix[3][1] = position.y;
	localToWorldMatrix[3][2] = position.z;
	MarkWorldTransformDirty();

	//glm::mat4 mat = glm::translate(localToWorldMatrix, position);
	//SetWorldTransform(mat);
}

void Node::SetPositon(glm::vec3&& position)
{
	SetPositon(position);
}

void Node::SetScale(glm::vec3& scale)
{
	glm::mat4 mat = glm::scale(localToWorldMatrix, scale);
	SetWorldTransform(mat);
}

void Node::SetScale(glm::vec3&& scale)
{
	SetScale(scale);
}

void Node::SetRotation(glm::vec3& eulerAngles)
{
	//glm::mat4_cast(glm::quat(glm::radians(eulerAngles)));
	glm::mat4 mat = glm::scale(localToWorldMatrix, glm::radians(eulerAngles));
	SetWorldTransform(mat);
}

void Node::SetRotation(glm::vec3&& eulerAngles)
{
	SetRotation(eulerAngles);
}

void Node::SetWorldTransform(glm::mat4& matrix)
{
	localToWorldMatrix = matrix;
	MarkWorldTransformDirty();
}

void Node::SetWorldTransform(glm::mat4&& matrix)
{
	SetWorldTransform(matrix);
}

void Node::MarkWorldTransformDirty()
{
	worldTransformDirty = true;
	for (Node* node : children)
	{
		node->MarkLocalTransformDirty();
	}
}

void Node::MarkLocalTransformDirty()
{
	localTransformDirty = true;
	for (Node* node : children)
	{
		node->MarkLocalTransformDirty();
	}
}

void Node::UpdateWorldTransformFromLoacl()
{
	if (parent != nullptr)
	{
		localToWorldMatrix = parent->localToWorldMatrix * localTransform;
	}
	else
	{
		localToWorldMatrix = localTransform;
	}

	localTransformDirty = false;
}

void Node::UpdateLocalTransformFromWorld()
{
	if (parent != nullptr)
	{
		localTransform = glm::inverse(parent->localToWorldMatrix) * localToWorldMatrix;
	}
	else
	{
		localTransform = localToWorldMatrix;
	}

	worldTransformDirty = false;
}
