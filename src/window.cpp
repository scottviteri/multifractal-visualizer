#include <glad/glad.h>
#include <GLFW/glfw3.h>
int main() {
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(800, 600, "GLAD Test", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGL();
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}
