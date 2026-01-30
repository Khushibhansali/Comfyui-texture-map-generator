# ComfyUI Texture Map Generator (Prompt → PBR Set)

This repository provides a custom ComfyUI node that generates a full Physically Based Rendering (PBR) texture set from a text prompt for use in 3D content-creation workflows.

## What PBR Means (Conceptual Overview)

Physically Based Rendering (PBR) is a workflow for shading and texturing that aims to produce consistent, realistic material appearance under different lighting environments. Instead of “baking” lighting into a single image, PBR represents a surface using multiple texture maps that each describe a specific physical property of the material. A renderer combines these maps with the scene’s lights and environment to compute the final look.

Common PBR maps:

- **Albedo (Base Color):** The surface’s intrinsic color, excluding lighting and shadow. No highlights or shading should be painted into albedo for a standard PBR workflow.
- **Normal:** Encodes small-scale surface direction changes to create the appearance of fine detail without increasing geometry. Used for lighting response.
- **Roughness:** Controls the spread of specular reflections. Lower roughness produces sharper reflections; higher roughness produces blurrier reflections.
- **Metallic:** Indicates whether the surface behaves like a metal. Metals typically have metallic=1 and use albedo as the specular color; non-metals typically have metallic=0.
- **Ambient Occlusion (AO):** Approximates how much ambient light is blocked in crevices and cavities. Often multiplied with albedo or used as an input where supported.
- **Displacement (Height):** Encodes height variation. Can be used for parallax/relief mapping or true geometric displacement depending on the renderer and asset pipeline.

These maps are commonly used in engines and tools such as Blender, Unreal Engine, Unity, Substance-based workflows, and physically-based offline renderers.

## What This Node Does

The node takes a text prompt and outputs a set of texture maps:

- albedo
- normal
- roughness
- metallic
- ambient occlusion
- displacement (height)

The implementation uses a text-to-material generation backend and converts the generated maps into ComfyUI `IMAGE` outputs so they can be routed into downstream nodes (preview, upscale, save to disk, packing).

## Repository Contents

- `__init__.py`  
  Registers the node with ComfyUI by exporting `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.

- `pbr.py`  
  Node implementation. Defines the node class, inputs, outputs, and the generation/inference logic.

- `requirements.txt`  
  Python dependencies required by the node.

- `example_workflow.json`
  A sample ComfyUI workflow that wires the node outputs to Save Image nodes.

## Installation

### ComfyUI Desktop (Windows)

ComfyUI Desktop runs ComfyUI from a separate base directory (the “basePath”), which is where `custom_nodes` and the Python environment are located.

1. Locate the ComfyUI basePath:
   - Open `%APPDATA%\ComfyUI\config.json`
   - Find `basePath`

2. Copy this repository folder into:
   - `<basePath>\custom_nodes\comfyui-texture-map-generator\`

3. Install dependencies into the ComfyUI Desktop Python environment:
   - Open a terminal in `<basePath>`
   - Run:
     ```bat
     .\.venv\Scripts\python.exe -m pip install -r custom_nodes\comfyui-texture-map-generator\requirements.txt
     ```

4. Restart ComfyUI Desktop.

### Standard ComfyUI (Git clone / portable)

1. Copy this repository folder into:
   - `ComfyUI/custom_nodes/comfyui-texture-map-generator/`

2. Install dependencies into the Python environment you use to launch ComfyUI:
   ```bash
   pip install -r custom_nodes/comfyui-texture-map-generator/requirements.txt
