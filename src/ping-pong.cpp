#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

const char* computeShaderSrc = R"(
#version 430
layout(local_size_x = 1) in;
layout(std430, binding = 0) readonly buffer Input {
    float inputData[];
};
layout(std430, binding = 1) buffer Output {
    float outputData[];
};
void main() {
    uint idx = gl_GlobalInvocationID.x;
    outputData[idx] = inputData[idx] * 2.0;
}
)";

int main() {
    // Initialize GLFW and OpenGL
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow(1, 1, "Ping-Pong Compute", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGL();

    // Compile compute shader
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(computeShader, 1, &computeShaderSrc, NULL);
    glCompileShader(computeShader);
    GLuint program = glCreateProgram();
    glAttachShader(program, computeShader);
    glLinkProgram(program);
    glDeleteShader(computeShader);

    // Initialize two buffers
    float data[10];
    for (int i = 0; i < 10; i++) data[i] = 1.0f;
    GLuint buffers[2];
    glGenBuffers(2, buffers);

    // Buffer 0: Initial data
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data), data, GL_DYNAMIC_COPY);

    // Buffer 1: Starts empty
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data), NULL, GL_DYNAMIC_COPY);

    // Iterations
    const int iterations = 3;
    for (int iter = 0; iter < iterations; iter++) {
        // Bind buffers: input = buffers[iter % 2], output = buffers[(iter + 1) % 2]
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers[iter % 2]);       // Input
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers[(iter + 1) % 2]); // Output

        // Clear the output buffer before writing
        float zero[10] = {0};
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[(iter + 1) % 2]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(zero), zero);

        // Run compute shader
        glUseProgram(program);
        glDispatchCompute(10, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // Retrieve final result
    float result[10];
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[iterations % 2]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(result), result);

    // Print
    for (float val : result) std::cout << val << " ";
    std::cout << std::endl; // Outputs: 8 8 8 8 8 8 8 8 8 8

    // Cleanup
    glDeleteBuffers(2, buffers);
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
