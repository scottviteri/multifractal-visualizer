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
uniform float probability; // Probability of using second fractal vs Sierpinski Triangle
uniform int secondFractalType; // 0 = Barnsley Fern, 1 = Sierpinski Carpet
uniform uint patternNumerator; // Numerator for deterministic pattern (a)
uniform uint patternDenominator; // Denominator for deterministic pattern (b)

// Simple hash function for randomness
float hash(vec2 p) {
    p = fract(p * vec2(123.4, 789.6));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

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
        
        bool useSecondFractal;
        
        // Use deterministic pattern if pattern denominator > 0, otherwise use probabilistic choice
        if (patternDenominator > 0) {
            // Deterministic pattern - use remainder of iteration divided by denominator
            // If remainderValue < numerator, use second fractal, otherwise use first
            uint patternPosition = currentIteration % patternDenominator;
            useSecondFractal = patternPosition < patternNumerator;
        } else {
            // Keep the original probabilistic approach as fallback
            float randVal = hash(p + vec2(0.1 * currentIteration, 0.2 * currentIteration));
            useSecondFractal = randVal < probability;
        }
        
        if (useSecondFractal) {
            if (secondFractalType == 0) {
                // Barnsley Fern (4 transformations)
                // The fern uses different affine transformations with varying probabilities
                // For our system, we'll apply all 4 transformations to every point
                
                // Stem transformation
                vec2 p1 = vec2(0.0, 0.16) * p;
                uint x1 = uint(clamp(p1.x, 0.0, 0.999) * width);
                uint y1 = uint(clamp(p1.y, 0.0, 0.999) * height);
                uint idx1 = (y1 * width + x1) / 32;
                uint bit1 = (y1 * width + x1) % 32;
                atomicOr(outputBuffer[idx1], (1u << bit1));
                
                // Left leaflet transformation
                // x' = 0.85x + 0.04y, y' = -0.04x + 0.85y + 0.16
                vec2 p2 = vec2(0.85*p.x + 0.04*p.y, -0.04*p.x + 0.85*p.y + 0.16);
                uint x2 = uint(clamp(p2.x, 0.0, 0.999) * width);
                uint y2 = uint(clamp(p2.y, 0.0, 0.999) * height);
                uint idx2 = (y2 * width + x2) / 32;
                uint bit2 = (y2 * width + x2) % 32;
                atomicOr(outputBuffer[idx2], (1u << bit2));
                
                // Right leaflet transformation
                // x' = 0.2x - 0.26y, y' = 0.23x + 0.22y + 0.16
                vec2 p3 = vec2(0.2*p.x - 0.26*p.y, 0.23*p.x + 0.22*p.y + 0.16);
                uint x3 = uint(clamp(p3.x, 0.0, 0.999) * width);
                uint y3 = uint(clamp(p3.y, 0.0, 0.999) * height);
                uint idx3 = (y3 * width + x3) / 32;
                uint bit3 = (y3 * width + x3) % 32;
                atomicOr(outputBuffer[idx3], (1u << bit3));
                
                // Successive leaflets transformation
                // x' = -0.15x + 0.28y, y' = 0.26x + 0.24y + 0.44
                vec2 p4 = vec2(-0.15*p.x + 0.28*p.y, 0.26*p.x + 0.24*p.y + 0.44);
                uint x4 = uint(clamp(p4.x, 0.0, 0.999) * width);
                uint y4 = uint(clamp(p4.y, 0.0, 0.999) * height);
                uint idx4 = (y4 * width + x4) / 32;
                uint bit4 = (y4 * width + x4) % 32;
                atomicOr(outputBuffer[idx4], (1u << bit4));
            }
            else if (secondFractalType == 1) {
                // Sierpinski Carpet (8 transformations)
                float scale = 1.0/3.0; // Scale factor
                
                // Apply 8 transformations, one for each sub-square except the center
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        // Skip the center square
                        if (i == 1 && j == 1) continue;
                        
                        vec2 offset = vec2(float(i), float(j)) * scale;
                        vec2 p1 = scale * p + offset;
                        uint x1 = uint(clamp(p1.x, 0.0, 0.999) * width);
                        uint y1 = uint(clamp(p1.y, 0.0, 0.999) * height);
                        uint idx1 = (y1 * width + x1) / 32;
                        uint bit1 = (y1 * width + x1) % 32;
                        atomicOr(outputBuffer[idx1], (1u << bit1));
                    }
                }
            }
        } 
        else {
            // Sierpinski triangle (3 transformations)
            float scale = 0.5; // Scale factor
            
            // Bottom-left corner
            vec2 p1 = scale * p;
            uint x1 = uint(p1.x * width);
            uint y1 = uint(p1.y * height);
            uint idx1 = (y1 * width + x1) / 32;
            uint bit1 = (y1 * width + x1) % 32;
            atomicOr(outputBuffer[idx1], (1u << bit1));
            
            // Bottom-right corner
            vec2 p2 = scale * p + vec2(0.5, 0.0);
            uint x2 = uint(p2.x * width);
            uint y2 = uint(p2.y * height);
            uint idx2 = (y2 * width + x2) / 32;
            uint bit2 = (y2 * width + x2) % 32;
            atomicOr(outputBuffer[idx2], (1u << bit2));
            
            // Top corner
            vec2 p3 = scale * p + vec2(0.25, 0.433); // 0.433 = sqrt(3)/4 = height of equilateral triangle
            uint x3 = uint(p3.x * width);
            uint y3 = uint(p3.y * height);
            uint idx3 = (y3 * width + x3) / 32;
            uint bit3 = (y3 * width + x3) % 32;
            atomicOr(outputBuffer[idx3], (1u << bit3));
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

// Helper function to find greatest common divisor (GCD)
uint32_t gcd(uint32_t a, uint32_t b) {
    while (b != 0) {
        uint32_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Helper function to convert a float probability to a rational approximation (numerator/denominator)
// Returns pair of {numerator, denominator}
std::pair<uint32_t, uint32_t> rationalApproximation(float value, uint32_t maxDenominator = 32) {
    if (value == 0.0f) return {0, 1};
    if (value == 1.0f) return {1, 1};
    
    // For example, if value = 0.5, we want to return {1, 2}
    uint32_t numerator = std::round(value * maxDenominator);
    uint32_t denominator = maxDenominator;
    
    // Simplify the fraction
    uint32_t divisor = gcd(numerator, denominator);
    numerator /= divisor;
    denominator /= divisor;
    
    return {numerator, denominator};
}

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
    stbi_write_png(filename.c_str(), windowWidth, windowHeight, 4, 
                  flippedPixels.data(), windowWidth * 4);
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    bool displayOnlyFullCycles = true; // Always use cycles-only mode by default
    int iterationOffset = 0; // Default offset is 0
    
    // Check for command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--offset" || arg == "-o") {
            if (i + 1 < argc) {
                try {
                    iterationOffset = std::stoi(argv[i+1]);
                    i++; // Skip the next argument which is the offset value
                    std::cout << "Using iteration offset: " << iterationOffset << std::endl;
                } catch (std::exception& e) {
                    std::cerr << "Error parsing offset value, using default of 0" << std::endl;
                }
            }
        }
    }

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
    GLFWwindow* window = glfwCreateWindow(1024, 1024, "Multifractal Visualizer", NULL, NULL);
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

    // Main loop variables
    GLuint inputBuffer = bufferA;
    GLuint outputBuffer = bufferB;
    int numIterations = 100000; // Number of IFS iterations
    int currentIteration = 0;
    float theta = 0.0f; // Starting angle
    const float rotationSpeed = 0.01f; // Speed of rotation in radians per frame
    float probability = 0.0f; // Initial probability value
    const int cyclePeriod = 50000; // Increased cycle period for smoother transitions
    const int numSteps = 1000; // Increased number of steps for even finer probability variations
    const float maxProbability = 1.0f; // Maximum probability value (now set to 1.0 instead of 0.9)
    float previousProbability = -1.0f; // Track probability changes to reset board
    std::vector<uint32_t> allOnesData(bufferSize, 0xFFFFFFFF); // For resetting the board
    
    // Display control variables
    const int minIterationsBeforeDisplay = 8; // Minimum iterations to process before displaying
    int iterationsSinceReset = 0; // Counter for iterations since last probability change/reset
    
    // Pattern variables for deterministic mode
    uint32_t patternNumerator = 0;
    uint32_t patternDenominator = 0; // 0 means use probabilistic mode
    bool useDeterministicMode = true; // Set to true to use deterministic pattern
    
    // Set which fractal to use as the second shape (0 = Barnsley Fern, 1 = Sierpinski Carpet)
    int secondFractalType = 1; // Change this to switch between fractal types
    std::string secondFractalNameTitle = (secondFractalType == 0) ? "Fern" : "Carpet"; // Capitalized for title
    std::string secondFractalName = (secondFractalType == 0) ? "fern" : "carpet"; // Lowercase for filenames

    // Main rendering loop
    while (!glfwWindowShouldClose(window) && currentIteration < numIterations) {
        // Calculate probability based on cycle position (ramp up and down)
        int cyclePosition = currentIteration % cyclePeriod;
        int halfCycle = cyclePeriod / 2;
        
        if (cyclePosition < halfCycle) {
            // First half: ramp up from 0 to maxProbability
            probability = (float)cyclePosition / halfCycle * maxProbability;
        } else {
            // Second half: ramp down from maxProbability to 0
            probability = maxProbability - (float)(cyclePosition - halfCycle) / halfCycle * maxProbability;
        }
        
        // Quantize probability to specific steps for better visualization
        probability = round(probability * numSteps) / numSteps;
        
        // Check if probability has changed - if so, reset the board
        bool resetRequired = false;
        if (probability != previousProbability) {
            resetRequired = true;
            previousProbability = probability;
            iterationsSinceReset = 0; // Reset the counter when probability changes
            
            // Calculate rational approximation for deterministic pattern
            if (useDeterministicMode) {
                auto [num, denom] = rationalApproximation(probability);
                patternNumerator = num;
                patternDenominator = denom;
                
                // If probability is 0 or 1, no need for pattern
                if (probability == 0.0f || probability == 1.0f) {
                    patternDenominator = 0; // Disable pattern
                }
            } else {
                patternDenominator = 0; // Use probabilistic mode
            }
            
            // Reset input buffer to all ones
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputBuffer);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bufferSize * sizeof(uint32_t), allOnesData.data());
            
            // Update window title with current probability
            std::string title = "Multifractal Visualizer - Sierpinski ";
            title += secondFractalNameTitle + " (p = " + std::to_string(probability).substr(0, 4) + ")";
            glfwSetWindowTitle(window, title.c_str());
            
            // Only print for significant probability changes (multiples of 0.1)
            if (fabs(round(probability * 10) - probability * 10) < 0.001) {
                std::cout << "Probability: " << probability << " (" 
                          << patternNumerator << "/" << patternDenominator << ")" << std::endl;
            }
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
        
        // Skip IFS update on reset frames to ensure clean state is displayed
        if (!resetRequired) {
            // Bind buffers for IFS update
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inputBuffer);  // Input
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, outputBuffer); // Output

            // Run IFS update compute shader
            glUseProgram(ifsUpdateProgram);
            glUniform1ui(glGetUniformLocation(ifsUpdateProgram, "width"), width);
            glUniform1ui(glGetUniformLocation(ifsUpdateProgram, "height"), height);
            glUniform1ui(glGetUniformLocation(ifsUpdateProgram, "currentIteration"), currentIteration);
            glUniform1f(glGetUniformLocation(ifsUpdateProgram, "theta"), theta);
            glUniform1f(glGetUniformLocation(ifsUpdateProgram, "probability"), probability);
            glUniform1i(glGetUniformLocation(ifsUpdateProgram, "secondFractalType"), secondFractalType);
            glUniform1ui(glGetUniformLocation(ifsUpdateProgram, "patternNumerator"), patternNumerator);
            glUniform1ui(glGetUniformLocation(ifsUpdateProgram, "patternDenominator"), patternDenominator);
            glDispatchCompute(64, 64, 1); // 1024x1024 / (16x16) = 64x64 work groups
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
            checkGLError("IFS update");
        }
        
        // Determine if we should render this frame
        bool shouldRender = true;
        
        if (displayOnlyFullCycles && useDeterministicMode && patternDenominator > 0) {
            // Only render on iterations that are multiples of the pattern denominator plus offset
            shouldRender = ((currentIteration + iterationOffset) % patternDenominator == 0);
        }
        
        // Skip displaying when we've just reset the board to all white
        // but we still want to process the reset internally
        shouldRender = shouldRender && !resetRequired;
        
        // Only render if we've performed the minimum number of iterations since the last reset
        shouldRender = shouldRender && (iterationsSinceReset >= minIterationsBeforeDisplay);
        
        if (shouldRender) {
            // Copy the data directly from outputBuffer to texture
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // Unbind any PBO first
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, resetRequired ? inputBuffer : outputBuffer);
            glBindTexture(GL_TEXTURE_2D, texture);
            
            // Use temporary buffer for data transfer
            std::vector<uint32_t> tempData(bufferSize);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bufferSize * sizeof(uint32_t), tempData.data());
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texWidth, texHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, tempData.data());
            checkGLError("Direct texture update");
            
            // Make sure texture is updated before rendering
            glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
            
            // Render the texture to screen
            glUseProgram(renderProgram);
            glUniform1ui(glGetUniformLocation(renderProgram, "width"), width);
            glUniform1ui(glGetUniformLocation(renderProgram, "height"), height);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glUniform1i(glGetUniformLocation(renderProgram, "tex"), 0);
            glBindVertexArray(quadVAO);
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
            
            // Get the name of the second fractal for file naming
            std::string secondFractalName = (secondFractalType == 0) ? "fern" : "carpet";
            
            // Save images only at key points in the probability cycle
            // Only save if we're not in a reset frame
            if (!resetRequired) {
                if (currentIteration == 0) {
                    // Initial state
                    saveFramebufferToImage(window, "output/initial_state.png");
                } else if (probability == 0.0f && previousProbability > 0.0f && currentIteration > cyclePeriod / 2) {
                    // End of cycle (after ramp down)
                    saveFramebufferToImage(window, "output/cycle_complete_" + secondFractalName + ".png");
                } else if (probability == maxProbability && previousProbability < maxProbability) {
                    // Peak of the cycle (maximum probability)
                    saveFramebufferToImage(window, "output/peak_p" + std::to_string(int(maxProbability * 100)) + "_" + secondFractalName + ".png");
                } else if (currentIteration == 500) {
                    // Midpoint check
                    saveFramebufferToImage(window, "output/midpoint_" + secondFractalName + "_500.png");
                } else if (currentIteration == 2000) {
                    // Later stage
                    saveFramebufferToImage(window, "output/later_stage_" + secondFractalName + "_2000.png");
                }
            }
            
            // Swap buffers and poll events
            glfwSwapBuffers(window);
        } else {
            // For frames we don't render, poll events to keep the window responsive
            glfwPollEvents();
        }
        
        // Only swap buffers if we didn't reset
        if (!resetRequired) {
            // Swap buffers for ping-pong - output becomes input for next iteration
            std::swap(inputBuffer, outputBuffer);
            
            // Increment iterations counter if we're in a post-reset state
            iterationsSinceReset++;
        }

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
