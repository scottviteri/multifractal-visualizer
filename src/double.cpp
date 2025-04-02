#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

// Compute shader source code
const char* computeShaderSrc = R"(
#version 430
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Data {
    float data[];
};
void main() {
    uint idx = gl_GlobalInvocationID.x;
    data[idx] *= 2.0;
}
)";

int main() {
    // Initialize GLFW and create a minimal window
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow(1, 1, "Compute Shader", NULL, NULL);
    glfwMakeContextCurrent(window);

    // Load OpenGL functions with GLAD
    gladLoadGL();

    // Create and compile the compute shader
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(computeShader, 1, &computeShaderSrc, NULL);
    glCompileShader(computeShader);

    // Create and link the shader program
    GLuint program = glCreateProgram();
    glAttachShader(program, computeShader);
    glLinkProgram(program);
    glDeleteShader(computeShader);

    // Initialize data and create a buffer
    float data[10];
    for (int i = 0; i < 10; i++) data[i] = 1.0f;
    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data), data, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);

    // Run the compute shader
    glUseProgram(program);
    glDispatchCompute(10, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Retrieve and print the results
    float result[10];
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(result), result);
    for (float val : result) std::cout << val << " ";
    std::cout << std::endl;

    // Clean up resources
    glDeleteBuffers(1, &buffer);
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
