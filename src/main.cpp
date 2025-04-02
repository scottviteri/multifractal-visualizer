#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

// Define STB_IMAGE_WRITE_IMPLEMENTATION before including to create the implementation
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // Make sure this header is in your include path

// Compute shader for IFS update (Different fractal transformations separated by a rotating line)
const char* ifsUpdateSrc = R"(
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer Input {
    uint inputBuffer[];
};
layout(std430, binding = 1) buffer Output {
    uint outputBuffer[];
};
uniform uint width;
uniform uint height;
uniform uint currentIteration;
uniform float theta; // Angle of the dividing line

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= width || y >= height) return;

    uint pixelIndex = y * width + x;
    uint uintIndex = pixelIndex / 32;
    uint bitIndex = pixelIndex % 32;
    uint mask = 1u << bitIndex;

    if ((inputBuffer[uintIndex] & mask) != 0) {
        vec2 p = vec2(float(x) / width, float(y) / height);
        
        // Convert coordinates to be centered at (0.5, 0.5)
        vec2 centered = p - vec2(0.5, 0.5);
        
        // Determine which side of the rotating line the point is on
        // Line direction vector is (cos(theta), sin(theta))
        // Use 2D cross product to determine side: centered.x * sin(theta) - centered.y * cos(theta)
        bool isRightSide = (centered.x * sin(theta) - centered.y * cos(theta)) > 0.0;
        
        if (isRightSide) {
            // Koch Curve (4 transformations) - right side of the line
            float scale = 0.33; // Scale factor
            
            // First segment
            vec2 p1 = scale * p;
            uint x1 = uint(p1.x * width);
            uint y1 = uint(p1.y * height);
            uint idx1 = (y1 * width + x1) / 32;
            uint bit1 = (y1 * width + x1) % 32;
            atomicOr(outputBuffer[idx1], (1u << bit1));
            
            // Second segment
            vec2 p2 = scale * p + vec2(scale, 0);
            uint x2 = uint(p2.x * width);
            uint y2 = uint(p2.y * height);
            uint idx2 = (y2 * width + x2) / 32;
            uint bit2 = (y2 * width + x2) % 32;
            atomicOr(outputBuffer[idx2], (1u << bit2));
            
            // Peak of the curve
            vec2 p3 = scale * p + vec2(0.5*scale, 0.866*scale);  // cos(60°), sin(60°)
            uint x3 = uint(p3.x * width);
            uint y3 = uint(p3.y * height);
            uint idx3 = (y3 * width + x3) / 32;
            uint bit3 = (y3 * width + x3) % 32;
            atomicOr(outputBuffer[idx3], (1u << bit3));
            
            // Third segment
            vec2 p4 = scale * p + vec2(2.0*scale, 0);
            uint x4 = uint(p4.x * width);
            uint y4 = uint(p4.y * height);
            uint idx4 = (y4 * width + x4) / 32;
            uint bit4 = (y4 * width + x4) % 32;
            atomicOr(outputBuffer[idx4], (1u << bit4));
        } 
        else {
            // Vicsek fractal (5 transformations) - left side of the line
            float scale = 0.33; // Scale factor
            
            // Center square
            vec2 p1 = scale * p + vec2(scale, scale);
            uint x1 = uint(p1.x * width);
            uint y1 = uint(p1.y * height);
            uint idx1 = (y1 * width + x1) / 32;
            uint bit1 = (y1 * width + x1) % 32;
            atomicOr(outputBuffer[idx1], (1u << bit1));
            
            // Top square
            vec2 p2 = scale * p + vec2(scale, 2.0*scale);
            uint x2 = uint(p2.x * width);
            uint y2 = uint(p2.y * height);
            uint idx2 = (y2 * width + x2) / 32;
            uint bit2 = (y2 * width + x2) % 32;
            atomicOr(outputBuffer[idx2], (1u << bit2));
            
            // Bottom square
            vec2 p3 = scale * p + vec2(scale, 0);
            uint x3 = uint(p3.x * width);
            uint y3 = uint(p3.y * height);
            uint idx3 = (y3 * width + x3) / 32;
            uint bit3 = (y3 * width + x3) % 32;
            atomicOr(outputBuffer[idx3], (1u << bit3));
            
            // Left square
            vec2 p4 = scale * p + vec2(0, scale);
            uint x4 = uint(p4.x * width);
            uint y4 = uint(p4.y * height);
            uint idx4 = (y4 * width + x4) / 32;
            uint bit4 = (y4 * width + x4) % 32;
            atomicOr(outputBuffer[idx4], (1u << bit4));
            
            // Right square
            vec2 p5 = scale * p + vec2(2.0*scale, scale);
            uint x5 = uint(p5.x * width);
            uint y5 = uint(p5.y * height);
            uint idx5 = (y5 * width + x5) / 32;
            uint bit5 = (y5 * width + x5) % 32;
            atomicOr(outputBuffer[idx5], (1u << bit5));
        }
    }
}
)";

// Compute shader for clearing the buffer
const char* clearSrc = R"(
#version 430
layout(local_size_x = 1024) in;
layout(std430, binding = 0) buffer Output {
    uint outputBuffer[];
};
uniform uint bufferSize;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < bufferSize) {
        outputBuffer[idx] = 0u;
    }
}
)";

// Vertex shader for rendering the buffer as a texture
const char* vertexSrc = R"(
#version 430
layout(location = 0) in vec2 position;
out vec2 texCoord;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    texCoord = (position + 1.0) * 0.5;
}
)";

// Fragment shader for rendering the buffer as a texture
const char* fragmentSrc = R"(
#version 430
in vec2 texCoord;
out vec4 color;
uniform usampler2D tex;
uniform uint width;
uniform uint height;

void main() {
    // Calculate the actual pixel coordinates
    ivec2 texSize = textureSize(tex, 0);
    uvec2 pixelCoord = uvec2(texCoord * vec2(width, height));
    
    // Calculate which 32-bit integer and which bit within it
    uint pixelIndex = pixelCoord.y * width + pixelCoord.x;
    uint intIndex = pixelIndex / 32u;
    uint bitIndex = pixelIndex % 32u;
    
    // Calculate texture coordinates in the packed texture
    uint texWidth = uint(texSize.x);
    ivec2 texCoords = ivec2(intIndex % texWidth, intIndex / texWidth);
    
    // Boundary check
    if(texCoords.x >= texSize.x || texCoords.y >= texSize.y) {
        color = vec4(1.0, 0.0, 0.0, 1.0); // Red for out of bounds
        return;
    }
    
    // Get the packed value from the texture
    uint packedValue = texelFetch(tex, texCoords, 0).r;
    
    // Use bit visualization
    uint mask = 1u << bitIndex;
    bool isSet = (packedValue & mask) != 0u;
    
    // Set colors based on bit value (white for set, black for unset)
    color = isSet ? vec4(1.0, 1.0, 1.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);
}
)";

// Helper function to compile a shader
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
    } else {
        std::cout << "Shader compiled successfully: " << (type == GL_VERTEX_SHADER ? "Vertex" : 
                                                          type == GL_FRAGMENT_SHADER ? "Fragment" : 
                                                          type == GL_COMPUTE_SHADER ? "Compute" : "Unknown") << std::endl;
    }
    return shader;
}

// Helper function to link a vertex and fragment shader into a program
GLuint linkProgram(GLuint vertex, GLuint fragment) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "Program linking error:\n" << infoLog << std::endl;
    } else {
        std::cout << "Program linked successfully" << std::endl;
    }
    return program;
}

// Helper function to check for OpenGL errors
void checkGLError(const char* operation) {
    GLenum error;
    while ((error = glGetError()) != GL_NO_ERROR) {
        std::cerr << "OpenGL error after " << operation << ": 0x" << std::hex << error << std::dec << std::endl;
    }
}

// Function to save the current framebuffer to a PNG file
void saveFramebufferToImage(GLFWwindow* window, const std::string& filename) {
    int windowWidth, windowHeight;
    glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
    
    // Allocate memory for the pixel data (RGBA format)
    std::vector<unsigned char> pixels(windowWidth * windowHeight * 4);
    
    // Read pixels from the framebuffer
    glReadPixels(0, 0, windowWidth, windowHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    
    // Flip the image vertically (OpenGL has the origin at the bottom-left)
    std::vector<unsigned char> flippedPixels(windowWidth * windowHeight * 4);
    for (int y = 0; y < windowHeight; y++) {
        for (int x = 0; x < windowWidth; x++) {
            for (int c = 0; c < 4; c++) {
                flippedPixels[(y * windowWidth + x) * 4 + c] = 
                    pixels[((windowHeight - 1 - y) * windowWidth + x) * 4 + c];
            }
        }
    }
    
    // Save the image using stb_image_write
    std::cout << "Saving image to " << filename << std::endl;
    int success = stbi_write_png(filename.c_str(), windowWidth, windowHeight, 4, 
                                flippedPixels.data(), windowWidth * 4);
    
    if (success) {
        std::cout << "Image saved successfully!" << std::endl;
    } else {
        std::cerr << "Failed to save image." << std::endl;
    }
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Configure GLFW for OpenGL 4.3 core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a 1024x1024 window
    GLFWwindow* window = glfwCreateWindow(1024, 1024, "Sierpinski Triangle", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Load OpenGL functions with GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Compile compute shaders
    GLuint ifsUpdateShader = compileShader(GL_COMPUTE_SHADER, ifsUpdateSrc);
    GLuint clearShader = compileShader(GL_COMPUTE_SHADER, clearSrc);
    GLuint ifsUpdateProgram = glCreateProgram();
    glAttachShader(ifsUpdateProgram, ifsUpdateShader);
    glLinkProgram(ifsUpdateProgram);
    GLuint clearProgram = glCreateProgram();
    glAttachShader(clearProgram, clearShader);
    glLinkProgram(clearProgram);

    // Compile rendering shaders
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);
    GLuint renderProgram = linkProgram(vertexShader, fragmentShader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Create a quad for rendering the texture
    float quadVertices[] = {
        -1.0f, -1.0f,  // Bottom-left
         1.0f, -1.0f,  // Bottom-right
         1.0f,  1.0f,  // Top-right
        -1.0f,  1.0f   // Top-left
    };
    GLuint quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Set up ping-pong buffers
    const uint32_t width = 1024, height = 1024;
    const uint32_t bufferSize = (width * height) / 32; // Number of 32-bit uints needed
    GLuint bufferA, bufferB;
    glGenBuffers(1, &bufferA);
    glGenBuffers(1, &bufferB);

    // Allocate memory for bufferA and bufferB
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferA);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize * sizeof(uint32_t), nullptr, GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferB);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize * sizeof(uint32_t), nullptr, GL_DYNAMIC_COPY);

    // Set up initial data - all bits set (all pixels on)
    std::vector<uint32_t> initialData(bufferSize, 0xFFFFFFFF); // Fill with all 1's (0xFFFFFFFF = 32 bits all set)
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferA);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bufferSize * sizeof(uint32_t), initialData.data());
    std::cout << "Initialized buffer with " << bufferSize << " integers (" << width*height << " bits) - all pixels on" << std::endl;

    // Create a texture to visualize the buffer
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    uint32_t texWidth = ceil(sqrt(bufferSize));
    uint32_t texHeight = ceil(float(bufferSize) / texWidth);

    // Initialize the texture directly with the buffer data for the first frame
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, texWidth, texHeight, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, initialData.data());
    std::cout << "Created texture: " << texWidth << "x" << texHeight << " (GL_R32UI)" << std::endl;

    // Main loop variables
    GLuint inputBuffer = bufferA;
    GLuint outputBuffer = bufferB;
    int numIterations = 10000; // Number of IFS iterations
    int currentIteration = 0;
    float theta = 0.0f; // Starting angle
    const float rotationSpeed = 0.01f; // Speed of rotation in radians per frame

    // Main rendering loop
    while (!glfwWindowShouldClose(window) && currentIteration < numIterations) {
        // Print every 100 iterations
        if (currentIteration % 100 == 0) {
            std::cout << "Iteration " << currentIteration << std::endl;
        }
        
        // Update the rotation angle
        theta += rotationSpeed;
        if (theta > 2 * M_PI) {
            theta -= 2 * M_PI; // Keep theta in [0, 2π] range
        }
        
        // Clear the output buffer before IFS update
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, outputBuffer);
        glUseProgram(clearProgram);
        glUniform1ui(glGetUniformLocation(clearProgram, "bufferSize"), bufferSize);
        glDispatchCompute(32, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        checkGLError("Clear buffer");
        
        // Bind buffers for IFS update
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inputBuffer);  // Input
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, outputBuffer); // Output

        // Run IFS update compute shader
        glUseProgram(ifsUpdateProgram);
        glUniform1ui(glGetUniformLocation(ifsUpdateProgram, "width"), width);
        glUniform1ui(glGetUniformLocation(ifsUpdateProgram, "height"), height);
        glUniform1ui(glGetUniformLocation(ifsUpdateProgram, "currentIteration"), currentIteration);
        glUniform1f(glGetUniformLocation(ifsUpdateProgram, "theta"), theta);
        glDispatchCompute(64, 64, 1); // 1024x1024 / (16x16) = 64x64 work groups
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        checkGLError("IFS update");

        // For debugging - print first few values of input and output
        if (currentIteration < 5) {
            std::vector<uint32_t> inputData(10);
            std::vector<uint32_t> outputData(10);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputBuffer);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 10 * sizeof(uint32_t), inputData.data());
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 10 * sizeof(uint32_t), outputData.data());
            
            std::cout << "Iteration " << currentIteration << " buffer comparison:" << std::endl;
            std::cout << "  Input[0-2]: " << std::hex << inputData[0] << " " << inputData[1] << " " << inputData[2] << std::dec << std::endl;
            std::cout << "  Output[0-2]: " << std::hex << outputData[0] << " " << outputData[1] << " " << outputData[2] << std::dec << std::endl;
        }
        
        // Copy the data directly from outputBuffer to texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // Unbind any PBO first
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
        glBindTexture(GL_TEXTURE_2D, texture);
        
        // Use temporary buffer for data transfer
        std::vector<uint32_t> tempData(bufferSize);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bufferSize * sizeof(uint32_t), tempData.data());
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texWidth, texHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, tempData.data());
        checkGLError("Direct texture update");
        
        // Make sure texture is updated before rendering
        glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
        
        // Swap buffers for ping-pong - output becomes input for next iteration
        std::swap(inputBuffer, outputBuffer);
        
        // Render the texture to screen
        glUseProgram(renderProgram);
        glUniform1ui(glGetUniformLocation(renderProgram, "width"), width);
        glUniform1ui(glGetUniformLocation(renderProgram, "height"), height);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(glGetUniformLocation(renderProgram, "tex"), 0);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

        // After setting uniforms, check their values
        if (currentIteration == 0) {
            GLuint widthLocation = glGetUniformLocation(renderProgram, "width");
            GLuint heightLocation = glGetUniformLocation(renderProgram, "height");
            GLuint texLocation = glGetUniformLocation(renderProgram, "tex");
            std::cout << "Uniform locations: width=" << widthLocation << ", height=" << heightLocation << ", tex=" << texLocation << std::endl;
        }

        // Save the image after 500 iterations
        if (currentIteration == 500) {
            saveFramebufferToImage(window, "rotating_vicsek_koch_500.png");
        }

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();

        currentIteration++;
    }

    // Cleanup OpenGL resources
    glDeleteBuffers(1, &bufferA);
    glDeleteBuffers(1, &bufferB);
    glDeleteProgram(ifsUpdateProgram);
    glDeleteProgram(clearProgram);
    glDeleteProgram(renderProgram);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteTextures(1, &texture);

    // Terminate GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
