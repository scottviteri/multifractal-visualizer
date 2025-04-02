#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <iostream>

// Compute shader source code
const char* computeShaderSource = R"(
#version 430
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Data {
    float data[];
};
void main() {
    uint index = gl_GlobalInvocationID.x;
    data[index] *= 2.0;
}
)";

// Function to compile a shader and check for errors
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader compilation error:\n" << infoLog << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// Function to link the shader program and check for errors
GLuint linkProgram(GLuint computeShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, computeShader);
    glLinkProgram(program);
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "Program linking error:\n" << infoLog << std::endl;
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Set OpenGL version to 4.3 core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a small window (1x1 pixel)
    GLFWwindow* window = glfwCreateWindow(1, 1, "Compute Shader Example", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Compile the compute shader
    GLuint computeShader = compileShader(GL_COMPUTE_SHADER, computeShaderSource);
    if (!computeShader) {
        glfwTerminate();
        return -1;
    }

    // Create and link the shader program
    GLuint program = linkProgram(computeShader);
    if (!program) {
        glDeleteShader(computeShader);
        glfwTerminate();
        return -1;
    }
    glDeleteShader(computeShader);

    // Create and initialize the buffer with data
    float data[10];
    for (int i = 0; i < 10; i++) data[i] = 1.0f;
    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data), data, GL_DYNAMIC_COPY);

    // Bind the buffer to binding point 0
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);

    // Use the compute shader program
    glUseProgram(program);

    // Dispatch the compute shader with 10 work groups
    glDispatchCompute(10, 1, 1);

    // Ensure memory operations are complete
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Read back the buffer
    float result[10];
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(result), result);

    // Print the results
    std::cout << "Results: ";
    for (int i = 0; i < 10; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    glDeleteBuffers(1, &buffer);
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}