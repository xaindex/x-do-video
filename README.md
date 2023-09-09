# X-Do Video

X-Do Video is a Xaindex generative model to create a video from a picture based on Stable Diffusion.

<img src="assets/inf_zooms/infinite_zoom_1.gif" width="200" height="200" /> <img src="assets/inf_zooms/infinite_zoom_2.gif" width="200" height="200" />
<img src="assets/inf_zooms/infinite_zoom_4.gif" width="200" height="200" /> <img src="assets/inf_zooms/infinite_zoom_3.gif" width="200" height="200" />

## Description

Given a prompt we run txt2img,py with [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
Then we paste a downscaled version of the image into its center and paint around the center using inpaint.py using this [sd-v1-5-inpainting.ckpt from](https://huggingface.co/runwayml/stable-diffusion-inpainting/tree/main) 
We repeated the inpainting step twice.

Then, zoom in by upscaling the image and cutting it to the original size while pasting the "center" image in its due area.


## How to run

### Download text-2-image and inpainting weights
hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5", filename="v1-5-pruned-emaonly.ckpt", cache_dir=".", use_auth_token=<HuggingFace token>)
hf_hub_download(repo_id="runwayml/stable-diffusion-inpainting", filename="sd-v1-5-inpainting.ckpt", cache_dir=".", use_auth_token=<HuggingFace token>)

### Create video
`python3 scripts/inf_zoom.py <your prompt>`
