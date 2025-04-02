CC = g++
CFLAGS = -std=c++11 -Wall -Wextra
INCLUDES = -I./include -I./external
LDFLAGS = -lGL -lm

# GLFW might be installed differently depending on the system
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LDFLAGS += -lglfw
endif
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -framework Cocoa -framework OpenGL -framework IOKit -lglfw
endif

# Source files
SOURCES = src/main.cpp src/glad.c

# Output binary
TARGET = multifractal_visualizer

# Build rule
all: ensure_directories $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

# Create directories if they don't exist
ensure_directories:
	@mkdir -p bin
	@mkdir -p include
	@mkdir -p external

# Run the application
run: all
	./$(TARGET)

# Download dependencies (for first-time setup)
setup:
	@echo "Downloading dependencies..."
	# Download stb_image_write.h if not present
	@if [ ! -f "include/stb_image_write.h" ]; then \
		echo "Downloading stb_image_write.h..."; \
		curl -o include/stb_image_write.h https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h; \
	fi
	# Download GLAD if not present
	@if [ ! -f "src/glad.c" ]; then \
		echo "Please visit https://glad.dav1d.de/ to generate GLAD for your system"; \
		echo "Select gl=4.3, Profile=Core, Generate a loader=checked"; \
		echo "Then place glad.c in src/ directory and glad/ folder in include/ directory"; \
	fi
	@echo "Setup complete. Please make sure GLFW is installed on your system."

# Clean build files
clean:
	rm -f $(TARGET)
	rm -rf *.png

.PHONY: all run clean setup ensure_directories 
