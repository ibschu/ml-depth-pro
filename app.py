import os
import gradio as gr
import requests
import json

# Check if the API token is set
print(f"REPLICATE_API_TOKEN is set: {'REPLICATE_API_TOKEN' in os.environ}")

# Replicate model identifier
MODEL = "chenxwh/ml-depth-pro:a6645b33f4e36eda0d8d52ab3da6ef37b82d198e2b70c72e680cc75f0baf1623"

def process_image(image_url):
    try:
        # Run the model using HTTP request
        print("Running the model...")
        try:
            headers = {
                "Authorization": f"Bearer {os.environ['REPLICATE_API_TOKEN']}",
                "Content-Type": "application/json",
                "Prefer": "wait"
            }
            data = {
                "version": "a6645b33f4e36eda0d8d52ab3da6ef37b82d198e2b70c72e680cc75f0baf1623",
                "input": {
                    "image_path": image_url
                }
            }
            response = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=data
            )
            
            if response.status_code not in [200, 201]:
                print(f"Replicate API request failed. Status code: {response.status_code}")
                print(f"Response content: {response.text}")
                return None, None, f"Replicate API request failed: {response.text}"
            
            output = response.json()
            
            # Convert the full output to a JSON string
            full_output_json = json.dumps(output, indent=2)
            
            print("Model run successful.")
            print("Full API Output:")
            print(full_output_json)
            
            color_map_url = output['output']['color_map']
            npz_url = output['output']['npz']
            
            print(f"Color map URL: {color_map_url}")
            print(f"NPZ URL: {npz_url}")
            
            return color_map_url, npz_url, "Success"
            
        except Exception as e:
            print(f"Replicate API error: {str(e)}")
            return None, None, f"Replicate API error: {str(e)}"
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None, f"Error: {str(e)}"

# Set up the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# ML-Depth-Pro Demo")
    gr.Markdown("The demo will process this image: https://www.strikeking.com/contentassets/4aba65e48f11486394435bfd5c8a334b/angler_holding_large_fish.jpg")
    
    process_btn = gr.Button("Process Image")
    
    color_map_url = gr.Textbox(label="Color Map URL")
    npz_url = gr.Textbox(label="NPZ URL")
    status = gr.Textbox(label="Status")

    process_btn.click(
        fn=lambda: process_image("https://www.strikeking.com/contentassets/4aba65e48f11486394435bfd5c8a334b/angler_holding_large_fish.jpg"),
        inputs=[],
        outputs=[color_map_url, npz_url, status]
    )

# Launch the Gradio interface
iface.launch()
