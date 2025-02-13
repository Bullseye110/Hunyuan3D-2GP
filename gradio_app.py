import os
import shutil
import time
from glob import glob
from pathlib import Path

import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# --- Monkey Patches for torch (must be applied before other imports) ---
if not hasattr(torch, "uint16"):
    torch.uint16 = torch.int16  # Using signed int16 as a placeholder
if not hasattr(torch, "uint32"):
    torch.uint32 = torch.int32  # Using signed int32 as a placeholder
if not hasattr(torch, "uint64"):
    torch.uint64 = torch.int64  # Using signed int64 as a placeholder

if not hasattr(torch.nn, "Buffer"):
    torch.nn.Buffer = lambda x: x
# --------------------------------------------------------------------

from mmgp import offload

# ====================================================
# NOTE on pymeshlab E57 Plugin Warning:
# ----------------------------------------------------
# The following warning may appear:
#
#   Warning:
#   Unable to load the following plugins:
#
#       libio_e57.so: libio_e57.so does not seem to be a Qt Plugin.
#
#   Cannot load library .../pymeshlab/lib/plugins/libio_e57.so: 
#   ... undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
#
# If you don't require E57 file support, you can safely ignore this warning.
# Alternatively, if you need E57 support, update your system's libffi and libp11-kit
# or manually remove/rename the file.
# ====================================================

def get_example_img_list():
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/*.png'))

def get_example_txt_list():
    print('Loading example txt list ...')
    txt_list = []
    with open('./assets/example_prompts.txt', encoding='utf-8') as f:
        for line in f:
            txt_list.append(line.strip())
    return txt_list

def gen_save_folder(max_size=60):
    os.makedirs(SAVE_DIR, exist_ok=True)
    exists = set(int(name) for name in os.listdir(SAVE_DIR) if name.isdigit())
    cur_id = min(set(range(max_size)) - exists) if len(exists) < max_size else -1
    folder_to_remove = f"{SAVE_DIR}/{(cur_id + 1) % max_size}"
    if os.path.exists(folder_to_remove):
        shutil.rmtree(folder_to_remove)
        print(f"Removed {folder_to_remove} successfully!")
    save_folder = f"{SAVE_DIR}/{max(0, cur_id)}"
    os.makedirs(save_folder, exist_ok=True)
    print(f"Created folder {save_folder} successfully!")
    return save_folder

def export_mesh(mesh, save_folder, textured=False):
    if textured:
        path = os.path.join(save_folder, 'textured_mesh.glb')
    else:
        path = os.path.join(save_folder, 'white_mesh.glb')
    mesh.export(path, include_normals=textured)
    return path

def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    if textured:
        related_path = "./textured_mesh.glb"
        template_name = './assets/modelviewer-textured-template.html'
        output_html_path = os.path.join(save_folder, 'textured_mesh.html')
    else:
        related_path = "./white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, 'white_mesh.html')

    # Read the template file from CURRENT_DIR
    with open(os.path.join(CURRENT_DIR, template_name), 'r', encoding='utf-8') as f:
        template_html = f.read()

    # Build the custom model-viewer block.
    obj_html = f"""
        <div class="column is-mobile is-centered">
            <model-viewer style="height: {height - 10}px; width: {width}px;" rotation-per-second="10deg" id="modelViewer"
                src="{related_path}/" disable-tap 
                environment-image="neutral" auto-rotate camera-target="0m 0m 0m" orientation="0deg 0deg 170deg" shadow-intensity=".9"
                ar auto-rotate camera-controls>
            </model-viewer>
        </div>
    """
    # Write the final HTML file to the save_folder.
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(template_html.replace('<model-viewer>', obj_html))
    
    # Compute the relative path from SAVE_DIR to the generated HTML file.
    # Using replace() ensures we get the correct relative URL.
    static_dir_abs = os.path.abspath(SAVE_DIR)
    output_html_abs = os.path.abspath(output_html_path)
    rel_path = os.path.normpath(output_html_abs.replace(static_dir_abs, "").lstrip(os.sep))
    # For debugging: print the computed relative path.
    print(f"Generated HTML file at: {output_html_abs}")
    print(f"Relative path for iframe: {rel_path}")
    
    # Build an iframe tag that points to the static file.
    iframe_tag = f'<iframe src="/static/{rel_path}" height="{height}" width="100%" frameborder="0"></iframe>'
    return f"""
        <div style='height: {height}; width: 100%;'>
            {iframe_tag}
        </div>
    """

def _gen_shape(caption, image, steps=50, guidance_scale=7.5, seed=1234,
               octree_resolution=256, check_box_rembg=False):
    if caption:
        print('Prompt:', caption)
    save_folder = gen_save_folder()
    stats = {}
    time_meta = {}
    start_time_0 = time.time()

    if image is None:
        start_time = time.time()
        try:
            image = t2i_worker(caption)
        except Exception as e:
            raise gr.Error("Text to 3D is disabled. Run with '--enable_t23d' to activate it.")
        time_meta['text2image'] = time.time() - start_time

    image.save(os.path.join(save_folder, 'input.png'))
    print("Image mode:", image.mode)
    if check_box_rembg or image.mode == "RGB":
        start_time = time.time()
        image = rmbg_worker(image.convert('RGB'))
        time_meta['rembg'] = time.time() - start_time
    image.save(os.path.join(save_folder, 'rembg.png'))

    start_time = time.time()
    generator = torch.Generator().manual_seed(int(seed))
    mesh = i23d_worker(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution
    )[0]

    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)

    stats['number_of_faces'] = mesh.faces.shape[0]
    stats['number_of_vertices'] = mesh.vertices.shape[0]
    time_meta['image_to_textured_3d'] = {'total': time.time() - start_time}
    time_meta['total'] = time.time() - start_time_0
    stats['time'] = time_meta
    return mesh, image, save_folder

def generation_all(caption, image, steps=50, guidance_scale=7.5, seed=1234,
                   octree_resolution=256, check_box_rembg=False):
    mesh, image, save_folder = _gen_shape(
        caption, image, steps, guidance_scale, seed, octree_resolution, check_box_rembg
    )
    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=596, width=700)
    textured_mesh = texgen_worker(mesh, image)
    path_textured = export_mesh(textured_mesh, save_folder, textured=True)
    model_viewer_html_textured = build_model_viewer_html(save_folder, height=596, width=700, textured=True)
    
    # Return file download links (which will not preview inline)
    # and the HTML preview code for each mesh.
    return (
        gr.update(value=path, visible=True),
        gr.update(value=path_textured, visible=True),
        model_viewer_html,
        model_viewer_html_textured,
    )

def shape_generation(caption, image, steps=50, guidance_scale=7.5, seed=1234,
                     octree_resolution=256, check_box_rembg=False):
    mesh, image, save_folder = _gen_shape(
        caption, image, steps, guidance_scale, seed, octree_resolution, check_box_rembg
    )
    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=596, width=700)
    return (
        gr.update(value=path, visible=True),
        model_viewer_html,
    )

def build_app():
    title_html = """
    <div align="center">
        <H1>Hunyuan3D-2<SUP>GP</SUP></H1>
        <BR>
        <B>Original model by Tencent, GPU Poor version by DeepBeepMeep. 
        Now this great 3D video generator can run smoothly with a 6 GB rig.</B>
        <BR><BR>
        Tencent Hunyuan3D Team - 
        <a href="https://github.com/tencent/Hunyuan3D-2">Github Page</a> &ensp;
        <a href="http://3d-models.hunyuan.tencent.com">Homepage</a> &ensp;
        <a href="#">Technical Report</a> &ensp;
        <a href="https://huggingface.co/Tencent/Hunyuan3D-2">Models</a>
    </div>
    """

    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.0') as demo:
        gr.HTML(title_html)
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs() as tabs_prompt:
                    with gr.Tab('Image Prompt', id='tab_img_prompt'):
                        image = gr.Image(label='Image', type='pil', image_mode='RGBA', height=290)
                        with gr.Row():
                            check_box_rembg = gr.Checkbox(value=True, label='Remove Background')
                    with gr.Tab('Text Prompt', id='tab_txt_prompt', visible=HAS_T2I):
                        caption = gr.Textbox(label='Text Prompt',
                                             placeholder='HunyuanDiT will be used to generate image.',
                                             info='Example: A 3D model of a cute cat, white background')
                with gr.Accordion('Advanced Options', open=False):
                    num_steps = gr.Slider(maximum=50, minimum=20, value=30, step=1, label='Inference Steps')
                    octree_resolution = gr.Dropdown([256, 384, 512], value=256, label='Octree Resolution')
                    cfg_scale = gr.Number(value=5.5, label='Guidance Scale')
                    seed = gr.Slider(maximum=1e7, minimum=0, value=1234, label='Seed')
                with gr.Group():
                    btn = gr.Button(value='Generate Shape Only', variant='primary')
                    btn_all = gr.Button(value='Generate Shape and Texture', variant='primary', visible=HAS_TEXTUREGEN)
                with gr.Group():
                    file_out = gr.File(label="File", visible=False)
                    file_out2 = gr.File(label="File", visible=False)
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab('Generated Mesh'):
                        html_output1 = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Generated Textured Mesh'):
                        html_output2 = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab('Image to 3D Gallery', id='tab_img_gallery'):
                        with gr.Row():
                            gr.Examples(examples=example_is, inputs=[image],
                                        label="Image Prompts", examples_per_page=18)
                    with gr.Tab('Text to 3D Gallery', id='tab_txt_gallery', visible=HAS_T2I):
                        with gr.Row():
                            gr.Examples(examples=example_ts, inputs=[caption],
                                        label="Text Prompts", examples_per_page=18)
        if not HAS_TEXTUREGEN:
            gr.HTML("""
            <div style="margin-top: 20px;">
                <b>Warning:</b> Texture synthesis is disabled due to missing requirements.
                Please install the requirements as described in README.md.
            </div>
            """)
        if not args.enable_t23d:
            gr.HTML("""
            <div style="margin-top: 20px;">
                <b>Warning:</b> Text to 3D is disabled. To enable it, run `python gradio_app.py --enable_t23d`.
            </div>
            """)
        # Set up interactions.
        btn.click(
            shape_generation,
            inputs=[caption, image, num_steps, cfg_scale, seed, octree_resolution, check_box_rembg],
            outputs=[file_out, html_output1]
        ).then(
            lambda: gr.update(visible=True),
            outputs=[file_out]
        )
        btn_all.click(
            generation_all,
            inputs=[caption, image, num_steps, cfg_scale, seed, octree_resolution, check_box_rembg],
            outputs=[file_out, file_out2, html_output1, html_output2]
        ).then(
            lambda: (gr.update(visible=True), gr.update(visible=True)),
            outputs=[file_out, file_out2]
        )
    return demo

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--cache-path', type=str, default='gradio_cache')
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--profile', type=str, default="3")
    parser.add_argument('--verbose', type=str, default="1")
    args = parser.parse_args()

    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    HTML_OUTPUT_PLACEHOLDER = """
    <div style='height: 596px; width: 100%; border-radius: 8px; border: 1px solid #e5e7eb;'></div>
    """
    INPUT_MESH_HTML = """
    <div style='height: 490px; width: 100%; border-radius: 8px; border: 1px solid #e5e7eb;'></div>
    """
    example_is = get_example_img_list()
    example_ts = get_example_txt_list()
    torch.set_default_device("cpu")

    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        texgen_worker = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
        HAS_TEXTUREGEN = True
    except Exception as e:
        print(e)
        print("Failed to load texture generator. Please check README.md for requirements.")
        HAS_TEXTUREGEN = False

    HAS_T2I = False
    if args.enable_t23d:
        from hy3dgen.text2image import HunyuanDiTPipeline
        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        HAS_T2I = True

    from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.rembg import BackgroundRemover

    rmbg_worker = BackgroundRemover()
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2', device="cpu", use_safetensors=True)
    FloaterRemover = FloaterRemover
    DegenerateFaceRemover = DegenerateFaceRemover
    FaceReducer = FaceReducer

    profile = int(args.profile)
    kwargs = {}
    pipe = offload.extract_models("i23d_worker", i23d_worker)
    if HAS_TEXTUREGEN:
        pipe.update(offload.extract_models("texgen_worker", texgen_worker))
        texgen_worker.models["multiview_model"].pipeline.vae.use_slicing = True
    if HAS_T2I:
        pipe.update(offload.extract_models("t2i_worker", t2i_worker))
    if profile < 5:
        kwargs["pinnedMemory"] = "i23d_worker/model"
    if profile != 1 and profile != 3:
        kwargs["budgets"] = {"*": 2200}
    offload.profile(pipe, profile_no=profile, verboseLevel=int(args.verbose), **kwargs)

    # Mount the SAVE_DIR as static files so generated HTML files can be served.
    app = FastAPI()
    static_dir = Path(SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

    demo = build_app()
    app = gr.mount_gradio_app(app, demo, path="/")
    demo.launch(share=True)
