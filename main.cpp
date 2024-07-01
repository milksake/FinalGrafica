// #define _GLIBCXX_DEBUG 1

#define GLAD_GL_IMPLEMENTATION
#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <utility>
#include <cmath>
#include <functional>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define WINDOW_WIDTH 800.0f
#define WINDOW_HEIGHT 600.0f
#define CAMERA_STEP 10.0f

const float PI = 3.1416f;
bool mode = false; // Geometry
bool fog = true;

int cameraZoom = 0;

// const float GLOBAL_SCALE = PI/10000.0f;
const float GLOBAL_SCALE = 0.001f;

// Shaders
const char* vertexShaderSource = R"glsl(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoord;
    layout (location = 2) in vec3 aNormal;
    layout (location = 3) in vec3 aAmb;
    layout (location = 4) in vec3 aDiff;
    layout (location = 5) in vec3 aSpec;
    layout (location = 6) in float aShin;
    layout (location = 7) in vec3 aColor;

    out vec2 TexCoord;
    out vec3 FragPos;
    out vec3 Normal;

    out vec3 mAmbient;
    out vec3 mDifusse;
    out vec3 mSpecular;
    out float mShininess;
    out vec3 mColor;
    out float distance;
    out vec3 proyVec;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    uniform vec3 viewPos;

    void main()
    {
        FragPos = vec3(model * vec4(aPos, 1.0));
        gl_Position = projection * view * vec4(FragPos, 1.0f);
        Normal = mat3(transpose(inverse(model))) * aNormal;

        TexCoord = aTexCoord;
        mAmbient = aAmb;
        mDifusse = aDiff;
        mSpecular = aSpec;
        mShininess = aShin;
        mColor = aColor;

        proyVec = viewPos - FragPos;
        distance = length(proyVec);
    }
)glsl";

const char* fragmentShaderSource = R"glsl(
    #version 330 core
    out vec4 FragColor;

    in vec2 TexCoord;
    in vec3 Normal;  
    in vec3 FragPos;

    in vec3 mAmbient;
    in vec3 mDifusse;
    in vec3 mSpecular;
    in float mShininess;
    in vec3 mColor;

    in float distance;
    in vec3 proyVec;
  
    uniform vec3 lightPos; 
    uniform vec3 viewPos; 
    uniform vec3 lightColor;

    uniform bool useTexture;
    uniform bool useFog;

    uniform sampler2D texture1;
    void main()
    {
        // Texture
        vec4 texColor;
        if (useTexture) {
            texColor = texture(texture1, TexCoord);
        } else {
            texColor = vec4(mColor, 1.0);
        }

        // ambient
        vec3 ambient = mAmbient * lightColor * 3.0;
        
        // diffuse 
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor * mDifusse;

        float attenuation = 1.0 / (1.0 + 0.001 * distance +  0.0001 * distance * distance);
        vec3 lightDir2 = normalize(proyVec);
        float diff2 = max(dot(norm, lightDir2), 0.0);
        diffuse += diff2 * vec3(1.0f, 1.0f, 1.0f) * mDifusse * attenuation;
        
        // specular
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec;
        if (mShininess == 0)
            spec = 1.0f;
        else
            spec = pow(max(dot(viewDir, reflectDir), 0.0), mShininess);
        vec3 specular = mSpecular * spec * lightColor; 

        vec3 result = (ambient + diffuse + specular) * vec3(texColor);

        if (useFog)
        {
            // fog
            vec4 fogColor = vec4(0.7, 0.7, 0.7, 1.0);
            float fogDensity = 0.7;
            float fogCoef = 1.0 / pow(2, distance*fogDensity * distance*fogDensity);
                
            FragColor = mix(fogColor, vec4(result, 1.0), fogCoef);
        }
        else
            FragColor = vec4(result, 1.0);
    }
)glsl";

const char *vertexShaderSource2 = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec2 aTexCoord;\n"
    "layout (location = 2) in vec3 aNormal;\n"
    "layout (location = 3) in vec3 aAmb;\n"
    "layout (location = 4) in vec3 aDiff;\n"
    "layout (location = 5) in vec3 aSpec;\n"
    "layout (location = 6) in float aShin;\n"
    "layout (location = 7) in vec3 aColor;\n"

    "uniform mat4 model;\n"
    "uniform mat4 view;\n"
    "uniform mat4 projection;\n"

    "uniform float scale;\n"
    "uniform float curv;\n"
    "uniform float anti;\n"

    "uniform vec3 viewPos;\n"

    "out vec2 TexCoord;\n"
    "out vec3 Normal;\n"
    "out vec3 FragPos;\n"
    
    "out vec3 mAmbient;\n"
    "out vec3 mDifusse;\n"
    "out vec3 mSpecular;\n"
    "out float mShininess;\n"
    "out vec3 mColor;\n"

    "out float distance;\n"
    "out vec3 proyVec;\n"

    "vec4 port(vec3 ePoint) // port from Euclidean geometry\n"
    "{\n"
    "	vec3 p = ePoint * scale; // scaling happens here\n"
    "	float d = length(p); // distance from geometry origin\n"
    "   vec4 outVec;\n"
    "	if (d < 0.00001f || curv == 0)\n"
    "       outVec = vec4(p, 1);\n"
    "   else\n"
    "	    outVec = vec4(p/d * sin(d), cos(d));\n"
    "   return outVec;\n"
    "}\n"
    
    "void main()\n"
    "{\n"
    "	vec4 newPos = model * vec4(aPos, 1.0f);\n"
    "   FragPos = newPos.xyz;\n"
    "   vec4 p = port(newPos.xyz);\n"
    "   gl_Position = projection * view * (anti * p);\n"
    "   proyVec = (gl_Position.xyz) / gl_Position.w;\n"
    "   float partial = dot(port(viewPos), p);\n"
    "   if (partial >= 0)\n"
    "       distance = 3.1416 / 2.0;\n"
    "   else\n"
    "       distance = acos(abs(partial));\n"
    "   Normal = anti * mat3(transpose(inverse(model))) * aNormal;\n"

    "	TexCoord = aTexCoord;\n"
    "   mAmbient = aAmb;\n"
    "   mDifusse = aDiff;\n"
    "   mSpecular = aSpec;\n"
    "   mShininess = aShin;\n"
    "   mColor = aColor;\n"
    "}\0";

const char *fragmentShaderSource2 = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D texture1;\n"
    "void main()\n"
    "{\n"
    "    FragColor = texture(texture1, TexCoord);\n"
    "}\0";

GLuint programs[2]; // 2 Geometries
class Camera* camera;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processKeyInput(GLFWwindow* window, int key, int scancode, int action, int mods);
GLuint loadShader(GLenum type, const char* source);
GLuint loadTexture(const std::string& path);

void printM(const glm::mat4x4& matrx)
{
    int n = 4;
	std::cout << "---\n";
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			std::cout << matrx[i][j] << ' ';
		}
		std::cout << '\n';
	}
	std::cout << "---\n";
}

glm::mat4x4 NonEuclideanTranslate(const glm::vec4& to)
{
    float LorentzSign = 1.0f;
    float denom = 1 + to.w;
    glm::mat4x4 out;
    out[0][0] = 1 - LorentzSign * to.x * to.x / denom; out[0][1] = -LorentzSign * to.x * to.y / denom; out[0][2] = -LorentzSign * to.x * to.z / denom; out[0][3] = -LorentzSign * to.x;
    out[1][0] = -LorentzSign * to.y * to.x / denom; out[1][1] = 1 - LorentzSign * to.y * to.y / denom; out[1][2] = -LorentzSign * to.y * to.z / denom; out[1][3] = -LorentzSign * to.y;
    out[2][0] = -LorentzSign * to.z * to.x / denom; out[2][1] = -LorentzSign * to.z * to.y / denom; out[2][2] = 1 - LorentzSign * to.z * to.z / denom; out[2][3] = -LorentzSign * to.z;
    out[3][0] = to.x; out[3][1] =  to.y; out[3][2] = to.z; out[3][3] = to.w;
    
    return glm::transpose(out);
}

std::string out;
class Model
{
    std::vector<float> vertices;
    std::vector<float> texcoords;
    std::vector<float> normals;
    std::vector<unsigned int> indices;
    std::vector<float> mAmb;
    std::vector<float> mDiff;
    std::vector<float> mSpec;
    std::vector<float> mShin;
    std::vector<float> mColor;
    GLuint vao, vbo, ebo;
    std::vector<GLuint> textureIDs;

    void setUpVao()
    {
        // std::cout << "Verticesss: " << vertices.size() << std::endl;
        // std::cout << "Indices: " << indices.size() << std::endl;

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);

        glBindVertexArray(vao);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        if (!texcoords.empty())
        {
            GLuint texVBO;
            glGenBuffers(1, &texVBO);
            glBindBuffer(GL_ARRAY_BUFFER, texVBO);
            glBufferData(GL_ARRAY_BUFFER, texcoords.size() * sizeof(float), texcoords.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(1);
        }
        if (!normals.empty())
        {
            GLuint normVBO;
            glGenBuffers(1, &normVBO);
            glBindBuffer(GL_ARRAY_BUFFER, normVBO);
            glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(2);
        }
        if (!mAmb.empty())
        {
            GLuint ambVBO;
            glGenBuffers(1, &ambVBO);
            glBindBuffer(GL_ARRAY_BUFFER, ambVBO);
            glBufferData(GL_ARRAY_BUFFER, mAmb.size() * sizeof(float), mAmb.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(3);
        }
        if (!mDiff.empty())
        {
            GLuint diffVBO;
            glGenBuffers(1, &diffVBO);
            glBindBuffer(GL_ARRAY_BUFFER, diffVBO);
            glBufferData(GL_ARRAY_BUFFER, mDiff.size() * sizeof(float), mDiff.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(4);
        }
        if (!mSpec.empty())
        {
            GLuint specVBO;
            glGenBuffers(1, &specVBO);
            glBindBuffer(GL_ARRAY_BUFFER, specVBO);
            glBufferData(GL_ARRAY_BUFFER, mSpec.size() * sizeof(float), mSpec.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(5);
        }
        if (!mShin.empty())
        {
            GLuint shinVBO;
            glGenBuffers(1, &shinVBO);
            glBindBuffer(GL_ARRAY_BUFFER, shinVBO);
            glBufferData(GL_ARRAY_BUFFER, mShin.size() * sizeof(float), mShin.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glEnableVertexAttribArray(6);
        }
        // else
        //     std::cout << "Empty--------------\n";
        if (!mColor.empty())
        {
            GLuint colorVBO;
            glGenBuffers(1, &colorVBO);
            glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
            glBufferData(GL_ARRAY_BUFFER, mColor.size() * sizeof(float), mColor.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(7);
        }

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        // std::cout << "\tAmbient: " << mAmb[0] << '\n';

        // vertices.clear();
        // normals.clear();
        // texcoords.clear();
        // mAmb.clear();
        // mDiff.clear();
        // mSpec.clear();
        // mShin.clear();
        // indices.clear();
    }

    bool loadModel(const std::string& path)
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        std::filesystem::path objPath = path;
        std::string baseDir = objPath.parent_path().string() + "/";

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str(), baseDir.c_str()))
        {
            std::cerr << "Error al cargar/parsear el archivo .obj: " << warn << err << std::endl;
            return false;
        }

        // std::cout << "Warning: " << warn << '\n';
        // std::cout << "Number of materials: " << materials.size() << '\n';

        // Loop over shapes
        for (size_t s = 0; s < shapes.size(); s++) {
            // Loop over faces(polygon)
            size_t index_offset = 0;
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

                // per-face material
                if (shapes[s].mesh.material_ids[f] >= materials.size() || shapes[s].mesh.material_ids[f] < 0)
                {
                    std::cerr << "Material inaccesible en: " << path << ", con indice: " << shapes[s].mesh.material_ids[f] << ", siendo los materials: " << materials.size() << std::endl;
                    exit(1000);
                }
                auto &mat = materials[shapes[s].mesh.material_ids[f]];

                // Loop over vertices in the face.
                for (size_t v = 0; v < fv; v++) {
                    // access to vertex
                    tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                    tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
                    tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
                    tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];

                    vertices.push_back(vx);
                    vertices.push_back(vy);
                    vertices.push_back(vz);

                    // Check if `normal_index` is zero or positive. negative = no normal data
                    if (idx.normal_index >= 0) {
                        tinyobj::real_t nx = attrib.normals[3*size_t(idx.normal_index)+0];
                        tinyobj::real_t ny = attrib.normals[3*size_t(idx.normal_index)+1];
                        tinyobj::real_t nz = attrib.normals[3*size_t(idx.normal_index)+2];

                        normals.push_back(nx);
                        normals.push_back(ny);
                        normals.push_back(nz);
                    }

                    // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                    if (true || idx.texcoord_index >= 0) {
                        tinyobj::real_t tx = attrib.texcoords[2*size_t(idx.texcoord_index)+0];
                        tinyobj::real_t ty = attrib.texcoords[2*size_t(idx.texcoord_index)+1];

                        texcoords.push_back(tx);
                        texcoords.push_back(ty);
                    }
                    // Optional: vertex colors
                    // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                    // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                    // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];

                    indices.push_back(indices.size());

                    mAmb.push_back(mat.ambient[0]);
                    mAmb.push_back(mat.ambient[1]);
                    mAmb.push_back(mat.ambient[2]);

                    mDiff.push_back(mat.diffuse[0]);
                    mDiff.push_back(mat.diffuse[1]);
                    mDiff.push_back(mat.diffuse[2]);

                    mSpec.push_back(mat.specular[0]);
                    mSpec.push_back(mat.specular[1]);
                    mSpec.push_back(mat.specular[2]);

                    mShin.push_back(mat.shininess);

                    mColor.push_back(mat.emission[0]);
                    mColor.push_back(mat.emission[1]);
                    mColor.push_back(mat.emission[2]);
                }
                index_offset += fv;
            }
        }

        // for (const auto& shape : shapes)
        // {
        //     for (const auto& index : shape.mesh.indices)
        //     {
        //         vertices.push_back(attrib.vertices[3 * index.vertex_index + 0]);
        //         vertices.push_back(attrib.vertices[3 * index.vertex_index + 1]);
        //         vertices.push_back(attrib.vertices[3 * index.vertex_index + 2]);
        //         if (!attrib.texcoords.empty()) {
        //             texcoords.push_back(attrib.texcoords[2 * index.texcoord_index + 0]);
        //             texcoords.push_back(attrib.texcoords[2 * index.texcoord_index + 1]);
        //         }
        //         indices.push_back(indices.size());
        //     }
        // }

        setUpVao();

        // carga las texturas desde el archivo .mtl
        for (const auto& material : materials)
        {
            if (!material.diffuse_texname.empty())
            {
                std::string texturePath = baseDir + material.diffuse_texname;
                textureIDs.push_back(loadTexture(texturePath));
            }
        }

        return true;
    }

public:
    Model(const std::string& path)
    {
        bool build = loadModel(path);
        if (!build)
            exit(1);
    }

    void draw(GLuint shaderProgram)
    {
        bool useTexture = !textureIDs.empty();
        glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), useTexture);

        if (useTexture)
        {
            for (size_t i = 0; i < textureIDs.size(); ++i)
            {
                glActiveTexture(GL_TEXTURE0 + i);
                glBindTexture(GL_TEXTURE_2D, textureIDs[i]);
                std::string uniformName = "texture" + std::to_string(i + 1);
                glUniform1i(glGetUniformLocation(shaderProgram, uniformName.c_str()), i);
            }
        }
        glBindVertexArray(vao);
		
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
};


class Object
{
    glm::vec4 position;
    glm::mat4x4 transformation;
    Model* model;
    glm::vec3 color;
    bool useColor;

public:
    Object(Model* _model, const glm::mat4x4& _transformation, const glm::vec3& _color = glm::vec3(1.0f), bool _useColor = false) :
        transformation(_transformation), model(_model), color(_color), useColor(_useColor)
    {
        position = transformation * glm::vec4(0.0f);
    }

    void draw(GLuint shaderProgram)
    {
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(transformation));
        if (useColor)
        {
            glUniform3fv(glGetUniformLocation(shaderProgram, "objectColor"), 1, glm::value_ptr(color));
        }
        model->draw(shaderProgram);
    }
};

glm::vec4 portEucToCurved(glm::vec4 eucPoint)
{
	glm::vec3 P = eucPoint;
	float distance = P.length();
	if (distance < 0.0001f) return eucPoint;
	/*if (LorentzSign > 0)*/ return glm::vec4(P / distance * std::sin(distance), std::cos(distance));
	// if (LorentzSign < 0) return float4(P / distance * sinh(distance), cosh(distance));
	return eucPoint;
}

class Camera
{
    glm::vec3 position;
    glm::vec3 center;

    glm::mat4x4 viewMatrix;
    glm::mat4x4 projMatrix;

    float fovy, aspect, near, far;

    void updateViewMatrix()
    {
        glUseProgram(programs[mode]);
        viewMatrix = glm::lookAt(position, center, glm::vec3(0.0f, 0.1f, 0.0f));

        if (mode == 1)
        {
            auto tmp = viewMatrix;

            glm::vec4 ic = glm::vec4(tmp[0][0], tmp[1][0], tmp[2][0], 0.0f);
            glm::vec4 jc = glm::vec4(tmp[0][1], tmp[1][1], tmp[2][1], 0.0f);
            glm::vec4 kc = glm::vec4(tmp[0][2], tmp[1][2], tmp[2][2], 0.0f);
            
            glm::vec4 geomEye = portEucToCurved(glm::vec4(position * GLOBAL_SCALE, 1.0f));

            glm::mat4x4 eyeTranslate = NonEuclideanTranslate(geomEye);
            // eyeTranslate[3][3] = geomEye[3];
            glm::vec4 icp, jcp, kcp;
            icp = eyeTranslate * ic;
            jcp = eyeTranslate * jc;
            kcp = eyeTranslate * kc;
            
            viewMatrix[0][0] = icp[0]; viewMatrix[0][1] = jcp[0]; viewMatrix[0][2] = kcp[0]; viewMatrix[0][3] = geomEye[0];
            viewMatrix[1][0] = icp[1]; viewMatrix[1][1] = jcp[1]; viewMatrix[1][2] = kcp[1]; viewMatrix[1][3] = geomEye[1];
            viewMatrix[2][0] = icp[2]; viewMatrix[2][1] = jcp[2]; viewMatrix[2][2] = kcp[2]; viewMatrix[2][3] = geomEye[2];
            viewMatrix[3][0] = icp[3]; viewMatrix[3][1] = jcp[3]; viewMatrix[3][2] = kcp[3]; viewMatrix[3][3] = geomEye[3];
        
            // viewMatrix =  glm::transpose(viewMatrix);
        }

        // std::cout << "-----Begin View Matrix\n";
		// printM(viewMatrix);
		// std::cout << "-----End View Matrix\n";
        // std::cout << "-----Begin View Raw\n";
        // auto rawww = glm::value_ptr(viewMatrix);
        // for (int i = 0; i < 16; i++)
        //     std::cout << rawww[i] << ' ';
        // std::cout << '\n';
		// std::cout << "-----End View Raw\n";

        glUniformMatrix4fv(glGetUniformLocation(programs[mode], "view"), 1, GL_FALSE, glm::value_ptr(viewMatrix));
        glUniform3f(glGetUniformLocation(programs[mode], "viewPos"), position[0], position[1], position[2]);
    }

    void updateProjectionMatrix()
    {
        glUseProgram(programs[mode]);
        if (mode == 0)
        {
            projMatrix = glm::perspective(fovy, aspect, near, far);
        }
        else
        {
            float sFovX = 1.0f/std::tan(fovy/2);
            float sFovY = 1.0f/std::tan(fovy * aspect/2);
            float div = (std::sin(far * GLOBAL_SCALE - near * GLOBAL_SCALE));
            float fp = (2.0f * std::sin(near * GLOBAL_SCALE) * std::sin(far * GLOBAL_SCALE)) / div;
            // float alph = (std::sin(near * GLOBAL_SCALE + far * GLOBAL_SCALE)) / div;
            float alph = 0;

            // sFovX, 0, 0, 0,
            // 0, sFovY, 0, 0,
            // 0, 0, -alph, -1,
            // 0, 0, -fp, 0

            projMatrix[0][0] = sFovX;
            projMatrix[1][1] = sFovY;
            projMatrix[2][2] = -alph;
            projMatrix[2][3] = -1;
            projMatrix[3][2] = -fp;
            
            // projMatrix = glm::transpose(projMatrix);
        }

        // std::cout << "-----Begin Projection Matrix\n";
		// printM(projMatrix);
		// std::cout << "-----End Projection Matrix\n";
		// std::cout << "-----Begin Projection Raw\n";
        // auto rawww = glm::value_ptr(projMatrix);
        // for (int i = 0; i < 16; i++)
        //     std::cout << rawww[i] << ' ';
        // std::cout << '\n';
		// std::cout << "-----End Projection Raw\n";
        
        glUniformMatrix4fv(glGetUniformLocation(programs[mode], "projection"), 1, GL_FALSE, glm::value_ptr(projMatrix));
    }

public:
    Camera(const glm::vec3& _position, const glm::vec3& _center, float _fovy, float _aspect, float _near, float _far) :
        position(_position), center(_center), fovy(_fovy), aspect(_aspect), near(_near), far(_far)
    {
        updateViewMatrix();
        updateProjectionMatrix();
    }

    void move(const glm::vec3& amount)
    {
        center += amount;
        position += amount;

        updateViewMatrix();
    }

    void turn(const glm::vec3& amount)
    {
        center += amount;

        updateViewMatrix();
    }

    void reset(const glm::vec3& _pos, const glm::vec3& _center)
    {
        center = _center;
        position = _pos;

        updateViewMatrix();
    }

    void update()
    {
        updateViewMatrix();
        updateProjectionMatrix();
    }

    glm::vec3 getCenter()
    {
        return center;
    }

    glm::vec3 getPosition()
    {
        return position;
    }
};

class Animation
{
    int currAnimation;
    int currFrame;
    bool finished;
    std::vector<std::pair<std::function<void(void)>, int>> animations;

public:

    Animation(const std::vector<std::pair<std::function<void(void)>, int>>& anims) :
        animations(anims), currAnimation(0), currFrame(0), finished(false) {}

    bool update()
    {
        if (finished)
            return true;
        (animations[currAnimation].first)();
        if (++currFrame == animations[currAnimation].second)
        {
            if (++currAnimation == animations.size())
            {
                finished = true;
                return true;
            }
            currFrame = 0;
        }
        return false;
    }
};

glm::vec3 lightPosition = glm::vec3(0.0f, 1.0f, 0.0f);
float step = 0.0001f;
float timer = 0.0f;

Animation* animation = nullptr;

void animate()
{
    lightPosition.y = std::sin(timer)*100.0f;
    glUniform3f(glGetUniformLocation(programs[mode], "lightPos"), lightPosition.x, lightPosition.y, lightPosition.z);

    timer += step;

    if (animation)
    {
        if (animation->update())
        {
            delete animation;
            animation = nullptr;
        }
    }
}

int main()
{
    // Inicializar GLFW
    if (!glfwInit()) {
        std::cerr << "Error al inicializar GLFW" << std::endl;
        return -1;
    }

    // Crear ventana
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Non-Euclidean 3D scene", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Error al crear la ventana GLFW" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, processKeyInput);

    // Inicializar GLAD
    if (!gladLoadGL(glfwGetProcAddress))
    {
        std::cerr << "Error al inicializar GLAD.\n";
        glfwTerminate();
        return -1;
    }

    // Flipping the image
    stbi_set_flip_vertically_on_load(true);

    // Compilar shaders
    GLuint vertexShader = loadShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = loadShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    programs[0] = glCreateProgram();
    glAttachShader(programs[0], vertexShader);
    glAttachShader(programs[0], fragmentShader);
    glLinkProgram(programs[0]);

    // Comprobar errores de enlace
    int success;
    char infoLog[512];
    glGetProgramiv(programs[0], GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(programs[0], 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Segundo Shader NO EUCLIDEANO
    vertexShader = loadShader(GL_VERTEX_SHADER, vertexShaderSource2);
    fragmentShader = loadShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    programs[1] = glCreateProgram();
    glAttachShader(programs[1], vertexShader);
    glAttachShader(programs[1], fragmentShader);
    glLinkProgram(programs[1]);

    glGetProgramiv(programs[1], GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(programs[1], 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glUseProgram(programs[1]);

    // Uniforms
    glUniform1f(glGetUniformLocation(programs[1], "scale"), GLOBAL_SCALE);
    glUniform1f(glGetUniformLocation(programs[1], "curv"), 1.0f);
    glUniform3f(glGetUniformLocation(programs[1], "lightPos"), lightPosition.x, lightPosition.y, lightPosition.z);
    glUniform3f(glGetUniformLocation(programs[1], "lightColor"), 1.0f, 1.0f, 0.5f);
    glUniform1i(glGetUniformLocation(programs[1], "useFog"), true);

    glUseProgram(programs[0]);
    glUniform3f(glGetUniformLocation(programs[0], "lightPos"), lightPosition.x, lightPosition.y, lightPosition.z);
    glUniform3f(glGetUniformLocation(programs[0], "lightColor"), 1.0f, 1.0f, 0.5f);
    glUniform3f(glGetUniformLocation(programs[0], "lightColor"), 1.0f, 1.0f, 0.5f);
    glUniform1i(glGetUniformLocation(programs[0], "useFog"), true);

    // Relative Path
	std::filesystem::path p = std::filesystem::current_path();
	int levels_path = 1;
	std::filesystem::path p_current;
	p_current = p.parent_path();

	for (int i = 0; i < levels_path; i++)
	{
		p_current = p_current.parent_path();
	}

	std::string vs_path, fs_path;

	std::stringstream ss;
	ss << std::quoted(p_current.string());
	ss >> std::quoted(out);

    out += "\\glfw-master\\OwnProjects\\Project_15\\Models\\";
	// std::cout << "Assets path: " << out << "\n";

    // Cargar modelos
    std::vector<Model> models;
	models.push_back(Model(out + "untitled1.obj"));
    models.push_back(Model(out + "stylized_house_OBJ.obj"));
    models.push_back(Model(out + "10438_Circular_Grass_Patch_v1_iterations-2.obj"));
    models.push_back(Model(out + "Lowpoly_Fox.obj"));
    models.push_back(Model(out + "Pig.obj"));
	models.push_back(Model(out + "Horse.obj"));
	models.push_back(Model(out + "Mushrooms1.obj"));

    // Crear objetos
	std::vector<Object> objects;
    // float yOffset = 10.0f;
    float yOffset = 10.0f;
    float xOffset = 0.0f;
    float zOffset = 0.0f;

    objects.push_back(
        Object(&models[2], // Grass
            glm::scale(
                glm::rotate(
                    glm::translate(
                        glm::mat4x4(1.0f),
                        glm::vec3(0.0f + xOffset, -9.0f + yOffset, 0.0f + zOffset)
                    ),
                    glm::radians(-90.0f),
                    glm::vec3(1.0f, 0.0f, 0.0f)
                ),
                glm::vec3(20.0f, 20.0f, 0.9f)
            )
        )
    );
    objects.push_back(
        Object(&models[2], // Grass 2
            glm::scale(
                glm::rotate(
                    glm::translate(
                        // glm::scale(
                        //     glm::mat4x4(1.0f),
                        //     glm::vec3(1.0f, -1.0f, 1.0f)
                        // ),
                        glm::mat4x4(1.0f),
                        glm::vec3(0.0f + xOffset, -9.0f + yOffset + 0.0f, 0.0f + zOffset)
                    ),
                    glm::radians(-90.0f),
                    glm::vec3(1.0f, 0.0f, 0.0f)
                ),
                glm::vec3(20.0f, 20.0f, -0.9f)
            )
        )
    );

	objects.push_back(
		Object(&models[1], // House
			glm::scale(
				glm::translate(
					glm::mat4x4(1.0f),
					glm::vec3(0.0f + xOffset, 40.0f + yOffset, -50.0f + zOffset) 
				),
				glm::vec3(5.9f)
			)
		)
	);
	objects.push_back(
		Object(&models[0], // Tree1
			glm::scale(
				glm::translate(
					glm::mat4x4(1.0f),
					glm::vec3(-100.0f + xOffset, 0.0f + yOffset, 0.0f + zOffset)
				),
				glm::vec3(5.0f)
			)
		)
	);
	objects.push_back(
		Object(&models[0], // Tree2
			glm::scale(
				glm::translate(
					glm::mat4x4(1.0f),
					glm::vec3(50.0f + xOffset, 0.0f + yOffset, 0.0f + zOffset)
				),
				glm::vec3(5.0f)
			)
		)
	);
	objects.push_back(
		Object(&models[0], // Tree1
			glm::scale(
				glm::translate(
					glm::mat4x4(1.0f),
					glm::vec3(-100.0f + xOffset, 0.0f + yOffset, -40.0f + zOffset)
				),
				glm::vec3(5.0f)
			)
		)
	);
	objects.push_back(
		Object(&models[0], // Tree1
			glm::scale(
				glm::translate(
					glm::mat4x4(1.0f),
					glm::vec3(-70.0f + xOffset, 0.0f + yOffset, -60.0f + zOffset)
				),
				glm::vec3(5.0f)
			)
		)
	);
	objects.push_back(
		Object(&models[0], // Tree1
			glm::scale(
				glm::translate(
					glm::mat4x4(1.0f),
					glm::vec3(50.0f + xOffset, 0.0f + yOffset, -70.0f + zOffset)
				),
				glm::vec3(5.0f)
			)
		)
	);
	objects.push_back(
		Object(&models[0], // Tree2
			glm::scale(
				glm::translate(
					glm::mat4x4(1.0f),
					glm::vec3(100.0f + xOffset, 0.0f + yOffset, -50.0f + zOffset)
				),
				glm::vec3(5.0f)
			)
		)
	);
	objects.push_back(
		Object(&models[0], // Tree2
			glm::scale(
				glm::translate(
					glm::mat4x4(1.0f),
					glm::vec3(60.0f + xOffset, 0.0f + yOffset, -90.0f + zOffset)
				),
				glm::vec3(5.0f)
			)
		)
	);

	objects.push_back(
		Object(&models[3], // Fox
			glm::scale(
				glm::translate(
					glm::mat4x4(1.0f),
					glm::vec3(0.0f + xOffset, 0.0f + yOffset, 50.0f + zOffset)
				),
				glm::vec3(0.1f)
			)
		)
	);
	objects.push_back(
		Object(&models[4], // Pig
			glm::scale(
				glm::translate(
					glm::mat4x4(1.0f),
					glm::vec3(60.0f + xOffset, 0.0f + yOffset, 50.0f + zOffset)
				),
				glm::vec3(0.1f)
			)
		)
	);
		objects.push_back(
		Object(&models[5], // Horse
			glm::scale(
				glm::translate(
					glm::mat4x4(1.0f),
					glm::vec3(80.0f + xOffset, 0.0f + yOffset, 50.0f + zOffset)
				),
				glm::vec3(5.0f)
			)
		)
	);
			objects.push_back(
		Object(&models[6], // hongos
			glm::scale(
				glm::translate(
					glm::mat4x4(1.0f),
					glm::vec3(-90.0f + xOffset, 2.0f + yOffset, 70.0f + zOffset)
				),
				glm::vec3(0.1f)
			)
		)
	);
				objects.push_back(
		Object(&models[6], // hongos
			glm::scale(
				glm::translate(
					glm::mat4x4(1.0f),
					glm::vec3(-50.0f + xOffset, 2.0f + yOffset, 50.0f + zOffset)
				),
				glm::vec3(0.1f)
			)
		)
	);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(83.0f/255.0f, 140.0f/255.0f, 22.0f/255.0f, 1.0f);
    glUseProgram(programs[mode]);

    camera = new Camera(
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, -1.0f),
        glm::radians(45.0f), WINDOW_WIDTH / WINDOW_HEIGHT, PI+1, 10000.0f
    );

    mode = !mode;
    camera->update();

    camera->move(27.0f * 10.0f * CAMERA_STEP * glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f)))));
    camera->turn(5.0f * -0.01f * CAMERA_STEP * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(1.0f, 0.0f, 0.0f))));
    // camera->move(glm::vec3(0.0f, 10.0f, 0.0f));

    std::cout << "Controls\n- Move camera: Arrow keys\n- Rotate Camera: WASD\n- Reset Camera: R\n- Animations: 1 2 3 keys\n- Toggle fog: F\n";

    // Bucle de renderizado
    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(programs[mode]);

        // Renderizar
        for (size_t i = 0; i < objects.size(); ++i)
        {
            animate();
            if (camera->getPosition().z >= -3200.0f)
            {  
                glUniform1i(glGetUniformLocation(programs[mode], "useFog"), false);
            }
            if (mode == 1)
            {
                glUniform1f(glGetUniformLocation(programs[1], "anti"), 1.0f);
            }
            objects[i].draw(programs[mode]);
            if (i == 0 || i == 1)
            {  
                glUniform1i(glGetUniformLocation(programs[mode], "useFog"), fog);
            }

        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void processKeyInput(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window, true);

    if (action == GLFW_PRESS && key == GLFW_KEY_F) // Fog
    {
        // std::cout << "Fog toggled\n";
        fog = !fog;
        glUniform1i(glGetUniformLocation(programs[mode], "useFog"), fog);
    }

    if (animation)
        return;

    if (action == GLFW_PRESS && key == GLFW_KEY_LEFT)
        camera->move(-5.0f * CAMERA_STEP * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f))));
    if (action == GLFW_PRESS && key == GLFW_KEY_RIGHT)
        camera->move(5.0f * CAMERA_STEP * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f))));
    if (action == GLFW_PRESS && key == GLFW_KEY_UP)
    {
        camera->move(10.0f * CAMERA_STEP * glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f)))));
        cameraZoom++;
    }
    if (action == GLFW_PRESS && key == GLFW_KEY_DOWN && cameraZoom)
    {
        camera->move(-10.0f * CAMERA_STEP * glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f)))));
        cameraZoom--;
    }
    if (action == GLFW_PRESS && key == GLFW_KEY_W)
        // camera->move(10.0f * CAMERA_STEP * glm::vec3(0.0f, 0.1f, 0.0f));
        camera->turn(-0.01f * CAMERA_STEP * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(1.0f, 0.0f, 0.0f))));
    if (action == GLFW_PRESS && key == GLFW_KEY_S)
        // camera->move(10.0f * CAMERA_STEP * glm::vec3(0.0f, -0.1f, 0.0f));
        camera->turn(0.01f * CAMERA_STEP * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(1.0f, 0.0f, 0.0f))));
    
    if (action == GLFW_PRESS && key == GLFW_KEY_A)
        camera->turn(-0.02f * CAMERA_STEP * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f))));
    if (action == GLFW_PRESS && key == GLFW_KEY_D)
        camera->turn(0.02f * CAMERA_STEP * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f))));


    if (action == GLFW_PRESS && key == GLFW_KEY_R)
    {
        camera->reset(
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, -1.0f)
        );
        camera->move(27.0f * 10.0f * CAMERA_STEP * glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f)))));
        camera->turn(5.0f * -0.01f * CAMERA_STEP * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(1.0f, 0.0f, 0.0f))));
        cameraZoom = 0;
    }

    if (action == GLFW_PRESS && key == GLFW_KEY_1)
    {
        animation = new Animation({
            {[](){
                camera->reset(
                    glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, -1.0f)
                );
                camera->move(27.0f * 10.0f * CAMERA_STEP * glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f)))));
                camera->turn(5.0f * -0.01f * CAMERA_STEP * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(1.0f, 0.0f, 0.0f))));
            }, 1},
            {[](){
                camera->move(0.1f * glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f)))));
            }, 15000},
            {[](){
                camera->turn(0.0001f * -1.0f * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(1.0f, 0.0f, 0.0f))));
            }, 1000},
            {[](){
                camera->turn(0.0001f * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f))));
            }, 10000},
            {[](){
                camera->turn(0.0001f * -1.0f * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f))));
            }, 20000},
            {[](){
                camera->turn(0.0001f * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f))));
            }, 10000},
            {[](){
                camera->turn(0.0001f * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(1.0f, 0.0f, 0.0f))));
            }, 1000},
            {[](){
                camera->move(0.1f * -1.0f * glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f)))));
            }, 15000}
        });
    }

    if (action == GLFW_PRESS && key == GLFW_KEY_2)
    {
        animation = new Animation({
            {[](){
                camera->reset(
                    glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, -1.0f)
                );
                camera->move(27.0f * 10.0f * CAMERA_STEP * glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f)))));
                camera->turn(5.0f * -0.01f * CAMERA_STEP * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(1.0f, 0.0f, 0.0f))));
            }, 1},
            {[](){
                camera->move(0.1f * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f))));
                camera->turn(0.00005f * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f))));
            }, 40000},
            {[](){
                camera->move(-0.1f * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f))));
                camera->turn(-0.00005f * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f))));
            }, 40000}
        });
    }

    if (action == GLFW_PRESS && key == GLFW_KEY_3)
    {
        animation = new Animation({
            {[](){
                camera->reset(
                    glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, -1.0f)
                );
                camera->move(16.0f * 10.0f * CAMERA_STEP * glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f)))));
                camera->turn(5.0f * -0.01f * CAMERA_STEP * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(1.0f, 0.0f, 0.0f))));
                camera->move(-10.0f * CAMERA_STEP * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f))));
                camera->turn(-0.1f * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f))));
            }, 1},
            {[](){
                camera->move(-0.05f * glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f)))));
            }, 15000},
            {[](){
                camera->move(0.05f * glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f)))));
            }, 15000},
            {[](){
                camera->reset(
                    glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, -1.0f)
                );
                camera->move(27.0f * 10.0f * CAMERA_STEP * glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(0.0f, 1.0f, 0.0f)))));
                camera->turn(5.0f * -0.01f * CAMERA_STEP * glm::normalize(glm::cross(camera->getCenter() - camera->getPosition(), glm::vec3(1.0f, 0.0f, 0.0f))));
            }, 1},
        });
    }

}

GLuint loadShader(GLenum type, const char* source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);


    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    return shader;
}

GLuint loadTexture(const std::string& path)
{
    GLuint textureID;
    glGenTextures(1, &textureID);

    int width, height, nrChannels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
    if (data) {
        GLenum format;
        if (nrChannels == 1)
            format = GL_RED;
        else if (nrChannels == 3)
            format = GL_RGB;
        else if (nrChannels == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    } else {
        std::cerr << "Error al cargar la textura: " << path << std::endl;
        stbi_image_free(data);
        exit(1);
    }

    return textureID;
}
