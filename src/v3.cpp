#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <random>

// Shader sources
const char* computeShaderSource = R"(
#version 430 core
layout(local_size_x = 1, local_size_y = 1) in;
layout(rgba32f, binding = 0) uniform image2D outputImage;

// Define the triangle vertices
const vec2 vertices[3] = {
    vec2(0.0, 0.87),   // top
    vec2(-0.5, -0.5),  // bottom left
    vec2(0.5, -0.5)    // bottom right
};

uniform int numIterations;
uniform vec2 currentPoint;

void main() {
    vec2 point = currentPoint;
    
    // Random vertex selection is done on CPU side for simplicity
    // We just compute one iteration step here
    int vertexIdx = int(gl_WorkGroupID.x) % 3;
    vec2 newPoint = mix(point, vertices[vertexIdx], 0.5); // Move halfway to chosen vertex
    
    // Store the point in the image
    ivec2 pixelCoord = ivec2(
        int((newPoint.x + 0.5) * imageSize(outputImage).x),
        int((newPoint.y + 0.5) * imageSize(outputImage).y)
    );
    
    if (pixelCoord.x >= 0 && pixelCoord.x < imageSize(outputImage).x && 
        pixelCoord.y >= 0 && pixelCoord.y < imageSize(outputImage).y) {
        // Set the pixel to white
        imageStore(outputImage, pixelCoord, vec4(1.0, 1.0, 1.0, 1.0));
    }
}
)";

const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec2 position;
layout (location = 1) in vec2 texCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    TexCoord = texCoord;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D textureMap;
void main() {
    FragColor = texture(textureMap, TexCoord);
}
)";

// Helper function to check shader compilation/linking errors
void checkShaderErrors(GLuint shader, const std::string& type) {
    GLint success;
    GLchar infoLog[1024];
    
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << std::endl;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << std::endl;
        }
    }
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    // Set OpenGL version and profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // Create window
    const unsigned int SCR_WIDTH = 800;
    const unsigned int SCR_HEIGHT = 800;
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Sierpinski Triangle - Compute Shader", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    
    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }
    
    // Create compute shader
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(computeShader, 1, &computeShaderSource, NULL);
    glCompileShader(computeShader);
    checkShaderErrors(computeShader, "COMPUTE");
    
    GLuint computeProgram = glCreateProgram();
    glAttachShader(computeProgram, computeShader);
    glLinkProgram(computeProgram);
    checkShaderErrors(computeProgram, "PROGRAM");
    
    // Create rendering shaders
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    checkShaderErrors(vertexShader, "VERTEX");
    
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    checkShaderErrors(fragmentShader, "FRAGMENT");
    
    GLuint renderProgram = glCreateProgram();
    glAttachShader(renderProgram, vertexShader);
    glAttachShader(renderProgram, fragmentShader);
    glLinkProgram(renderProgram);
    checkShaderErrors(renderProgram, "PROGRAM");
    
    // Create texture for compute shader to write to
    GLuint texture;
    glGenTextures(1, &texture);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    
    // Clear the texture with black
    float* clearData = new float[SCR_WIDTH * SCR_HEIGHT * 4]();  // Initialized to zeros
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGBA, GL_FLOAT, clearData);
    delete[] clearData;
    
    // Setup vertices for rendering the texture to the screen
    float quadVertices[] = {
        // positions        // texture coords
        -1.0f,  1.0f,      0.0f, 1.0f,   // top left
        -1.0f, -1.0f,      0.0f, 0.0f,   // bottom left
         1.0f, -1.0f,      1.0f, 0.0f,   // bottom right
        -1.0f,  1.0f,      0.0f, 1.0f,   // top left
         1.0f, -1.0f,      1.0f, 0.0f,   // bottom right
         1.0f,  1.0f,      1.0f, 1.0f    // top right
    };
    
    GLuint quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    
    // Setup random generator for vertex selection
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> vertexDist(0, 2);
    
    // Start with a random point inside the triangle
    float px = 0.0f;
    float py = 0.0f;
    
    const int ITERATIONS_PER_FRAME = 10000;
    
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Generate points using the compute shader
        glUseProgram(computeProgram);
        
        // For each iteration, select a random vertex and move halfway to it
        for (int i = 0; i < ITERATIONS_PER_FRAME; i++) {
            // Update current point
            glUniform2f(glGetUniformLocation(computeProgram, "currentPoint"), px, py);
            
            // Dispatch compute shader with workgroup size=1 and dispatch size equal to the number of points to generate
            glDispatchCompute(1, 1, 1);
            
            // Ensure compute shader has finished before reading data
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            
            // Choose a random vertex for the next iteration
            int nextVertex = vertexDist(gen);
            const float vertices[3][2] = {
                {0.0f, 0.87f},    // top
                {-0.5f, -0.5f},   // bottom left
                {0.5f, -0.5f}     // bottom right
            };
            
            // Move halfway to the chosen vertex
            px = (px + vertices[nextVertex][0]) * 0.5f;
            py = (py + vertices[nextVertex][1]) * 0.5f;
        }
        
        // Render the texture to the screen
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(renderProgram);
        glBindVertexArray(quadVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(glGetUniformLocation(renderProgram, "textureMap"), 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    // Cleanup
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteTextures(1, &texture);
    glDeleteProgram(computeProgram);
    glDeleteProgram(renderProgram);
    glDeleteShader(computeShader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    glfwTerminate();
    return 0;
} 