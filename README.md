# Multifractal Visualizer

A GPU-accelerated fractal visualization tool that showcases the interplay between different fractal systems separated by a rotating dividing line.

![Probabilistic Fractal Animation](output/rotating_vicsek_koch.gif)
*Animated visualization of the probabilistic transition between Sierpinski and Koch fractal systems*

## Features

- Real-time generation of fractals using OpenGL compute shaders
- Bit-packed representation for efficient memory usage
- Fractal system selection with two modes:
  - **Probabilistic Mode**: Random selection between fractal systems based on probability
  - **Deterministic Mode**: Orderly pattern of fractal selections based on rational approximation of probability
- Automatic image export at configurable iterations
- Full GPU acceleration for all fractal calculations

## Fractal Systems

This visualization combines two fractal systems with distinct phases of selection:
- **Sierpinski Triangle**: A classic self-similar triangular fractal pattern
- **Second Fractal**: Configurable between:
  - **Barnsley Fern**: A natural-looking fractal that resembles a fern leaf
  - **Sierpinski Carpet**: A square-based fractal with a recursive pattern of holes

The second fractal can be easily switched by changing the `secondFractalType` variable in the code:
```cpp
// Set which fractal to use as the second shape
// 0 = Barnsley Fern, 1 = Sierpinski Carpet
int secondFractalType = 1; // Change this value to switch
```

### Fractal Selection Modes

By default, the program uses a deterministic pattern for fractal selection. This can be controlled with the `useDeterministicMode` flag:

```cpp
// Set to true for deterministic pattern, false for probabilistic selection
bool useDeterministicMode = true;
```

#### Deterministic Mode

In deterministic mode, when probability p = a/b (a rational number):
- For each sequence of b iterations, exactly a iterations will use the second fractal
- The remaining (b-a) iterations will use the Sierpinski Triangle

For example, with p = 0.5:
- The pattern becomes [Triangle, SecondFractal, Triangle, SecondFractal, ...]
- This creates a perfect alternating pattern instead of random 50/50 selection

#### Probabilistic Mode

In the original probabilistic mode, each point has a random chance (based on probability) to use either fractal system.

### Visualization Cycle

The visualization implements a smooth probability cycle that creates a continuous transition between fractal systems:

- **Ramp Up Phase**: Probability gradually increases from 0 to 1.0 during the first half of the cycle
- **Ramp Down Phase**: Probability gradually decreases from 1.0 back to 0 during the second half

This creates a smooth transition between:
- Pure Sierpinski triangle (probability = 0.0)
- Mixed states with varying degrees of the second fractal (intermediate probabilities)
- Pure second fractal (probability = 1.0)
- Then back to pure Sierpinski triangle

The probability values are quantized to discrete steps (50 steps by default) to create visible transitions between different mixing ratios while still providing a very fine-grained progression of fractal evolution. Each time the probability value changes, the visualization resets to a fresh starting state with all pixels on, allowing each phase to develop independently from this initial state.

## Dependencies

- OpenGL 4.3+
- GLFW3
- GLAD
- stb_image_write.h (single-header library for PNG output)

## Building the Project

### First Time Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/scottviteri/multifractal-visualizer.git
   cd multifractal-visualizer
   ```

2. Run the setup script to download required dependencies:
   ```bash
   make setup
   ```
   
3. For GLAD (OpenGL loader), visit [https://glad.dav1d.de/](https://glad.dav1d.de/) and:
   - Select OpenGL 4.3 or higher
   - Select "Core" profile
   - Check "Generate a loader"
   - Click "Generate"
   - Download the zip file
   - Place `glad.c` in the `src/` directory
   - Place the `glad/` folder in the `include/` directory

4. Install GLFW:
   - **Linux (Ubuntu/Debian)**: `sudo apt-get install libglfw3-dev`
   - **macOS**: `brew install glfw`
   - **Windows**: Download from [GLFW website](https://www.glfw.org/download.html) or use MSYS2/MinGW

### Building

After setup, build the project with:

```bash
make
```

### Running

```bash
make run
```

Or directly:

```bash
./multifractal_visualizer
```

#### Running on NVIDIA GPU

If you have a system with both integrated and NVIDIA graphics (Optimus technology), you can use:

```bash
make run-prime
```

This will execute the application with `prime-run`, ensuring it runs on the dedicated NVIDIA GPU for better performance.

#### Command Line Options

- **Iteration Offset**: Control which specific iterations of the pattern cycle get displayed
  ```bash
  ./multifractal_visualizer --offset 3  # Display iterations at positions 3, 3+n, 3+2n, etc.
  ```
  ```bash
  ./multifractal_visualizer -o 2  # Short form
  ```

The program uses a "cycles-only" display mode by default, which:
- Only updates the display after completing a full pattern cycle
- For a pattern of period n, only iterations 0, n, 2n, 3n, etc. will be displayed
- Using an offset changes this to display iterations offset, n+offset, 2n+offset, etc.
- All computations still happen on every iteration, but the display is only updated at cycle boundaries
- This provides a clearer view of how the fractal evolves after each complete pattern cycle

## Repository Structure

```
fractal-visualizer/
├── src/                # Source code
│   ├── main.cpp        # Main application code
│   └── glad.c          # GLAD OpenGL loader (you need to add this)
├── include/            # Header files
│   ├── glad/           # GLAD headers (you need to add this)
│   └── stb_image_write.h # Image writing library
├── Makefile            # Build configuration
└── README.md           # This file
```

## GitHub Repository Guidelines

When committing this project to GitHub:

1. **Do include** the `include/stb_image_write.h` file as it's a single-header library that's easy to distribute
2. **Do not include** the GLAD files in your commits - each user should generate these for their specific system
3. Add the following to your `.gitignore`:
   ```
   multifractal_visualizer
   src/glad.c
   include/glad/
   ```

## Images
Sample fractal visualizations are included in the repository's `output` folder:
![Sierpinski-Carpet](output/sierpinski_triangle_carpet_500.png)
*Sierpinski triangle and carpet fractal systems at 500 iterations*
![Vicsek-Koch](output/vicsek_koch_500.png)
*Vicsek and Koch fractal systems at 500 iterations*

## License

[MIT License](LICENSE)

## Acknowledgments

- [GLFW](https://www.glfw.org/) for window creation and OpenGL context management
- [GLAD](https://github.com/Dav1dde/glad) for OpenGL function loading
- [stb](https://github.com/nothings/stb) for the image writing library 