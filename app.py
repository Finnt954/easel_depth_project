from flask import Flask, request
from PIL import Image
import time
import os
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MODELS = {
    "Depth Anything v2": "depth-anything/Depth-Anything-V2-Base-hf",
    "ZoeDepth": "Intel/zoedepth-nyu"
}

loaded_models = {}
for name, model_id in MODELS.items():
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()
    loaded_models[name] = (processor, model)
    print(f"Loaded {name}")

print("All models loaded.")

def apply_cleanup(depth_arr):
    """Apply threshold + median + gaussian cleanup"""
    # Threshold background
    depth_clean = depth_arr.copy()
    depth_clean[depth_clean < 0.03] = 0
    
    # Median filter to remove speckle
    mask = depth_clean > 0
    if mask.sum() > 0:
        depth_clean[mask] = median_filter(depth_clean[mask], size=5)
        # Gaussian smooth
        depth_clean[mask] = gaussian_filter(depth_clean[mask], sigma=1.5)
        # Normalize to 0.35-1.0
        dmin = depth_clean[mask].min()
        dmax = depth_clean[mask].max()
        depth_clean[mask] = 0.35 + (depth_clean[mask] - dmin) / (dmax - dmin) * 0.65
    
    return depth_clean

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Depth Model Tester</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: #1a1a2e;
                color: #eee;
                text-align: center;
                padding: 50px;
            }
            h1 { color: #e94560; }
            form {
                background: #16213e;
                padding: 30px;
                border-radius: 12px;
                display: inline-block;
            }
            input, button {
                padding: 10px 20px;
                margin: 10px;
                border-radius: 6px;
                border: none;
                font-size: 16px;
            }
            button {
                background: #e94560;
                color: white;
                cursor: pointer;
                font-weight: bold;
            }
            button:hover { background: #c23152; }
            .result-box {
                background: #16213e;
                padding: 15px;
                margin: 10px;
                border-radius: 12px;
                display: inline-block;
                vertical-align: top;
                width: 280px;
            }
            .result-box img { width: 100%; border-radius: 8px; }
            .grid { display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; }
            a { color: #e94560; }
            .metric { color: #0f3460; background: #eee; padding: 3px 6px; border-radius: 4px; display: inline-block; margin: 2px; font-size: 13px; }
            h2 { color: #e94560; margin-top: 40px; }
        </style>
    </head>
    <body>
        <h1>Depth Model Tester</h1>
        <form action="/run" method="post" enctype="multipart/form-data">
            <input type="file" name="image" required><br>
            <button type="submit">Compare All Models</button>
        </form>
    </body>
    </html>
    '''

@app.route('/run', methods=['POST'])
def run():
    file = request.files['image']
    filename = file.filename
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)
    
    image = Image.open(img_path).convert("RGB")
    
    results = {}
    cleanup_results = {}
    
    # Apple Depth Pro
    try:
        from depth_pro import create_model_and_transforms
        model_dp, transform = create_model_and_transforms(device="cpu")
        model_dp.eval()
        inputs = transform(image)
        start = time.time()
        with torch.no_grad():
            depth = model_dp.infer(inputs)['depth']
        elapsed = time.time() - start
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
        depth_arr = depth_norm.squeeze().cpu().numpy()
        out_path = os.path.join(OUTPUT_FOLDER, 'apple_depth_pro.png')
        plt.imsave(out_path, depth_arr, cmap='inferno')
        results['Apple Depth Pro'] = {'time': elapsed, 'cost': 'free (local)', 'img': 'apple_depth_pro.png'}
        
        # Cleanup version
        cleaned = apply_cleanup(depth_arr)
        clean_path = os.path.join(OUTPUT_FOLDER, 'apple_depth_pro_clean.png')
        plt.imsave(clean_path, cleaned, cmap='inferno')
        cleanup_results['Apple Depth Pro'] = {'img': 'apple_depth_pro_clean.png'}
    except Exception as e:
        print(f"Apple Depth Pro error: {e}")
    
    # Depth Anything v2 and ZoeDepth
    for name, (processor, model) in loaded_models.items():
        inputs = processor(images=image, return_tensors="pt")
        start = time.time()
        with torch.no_grad():
            depth = model(**inputs).predicted_depth
        elapsed = time.time() - start
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
        depth_arr = depth_norm.squeeze().cpu().numpy()
        safe_name = name.replace(" ", "_").lower()
        out_path = os.path.join(OUTPUT_FOLDER, f'{safe_name}.png')
        plt.imsave(out_path, depth_arr, cmap='inferno')
        results[name] = {'time': elapsed, 'cost': 'free (local)', 'img': f'{safe_name}.png'}
        
        # Cleanup version
        cleaned = apply_cleanup(depth_arr)
        clean_path = os.path.join(OUTPUT_FOLDER, f'{safe_name}_clean.png')
        plt.imsave(clean_path, cleaned, cmap='inferno')
        cleanup_results[name] = {'img': f'{safe_name}_clean.png'}
    
    # Gemini placeholder
    results['Gemini'] = {'time': 3.5, 'cost': '~$0.00002 (API)', 'img': None}
    cleanup_results['Gemini'] = {'img': None}
    
    # Build HTML
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Results - Depth Model Tester</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: #1a1a2e;
                color: #eee;
                text-align: center;
                padding: 30px;
            }
            h1, h2 { color: #e94560; }
            .grid { display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; }
            .result-box {
                background: #16213e;
                padding: 15px;
                border-radius: 12px;
                width: 280px;
            }
            .result-box img { width: 100%; border-radius: 8px; }
            .metric { color: #0f3460; background: #eee; padding: 3px 6px; border-radius: 4px; display: inline-block; margin: 2px; font-size: 13px; }
            a { color: #e94560; }
        </style>
    </head>
    <body>
        <h1>Model Comparison</h1>
        <div class="grid">
    '''
    
    for name, data in results.items():
        speed = f"{data['time']:.3f}s"
        cost = data['cost']
        if data['img']:
            html += f'''
            <div class="result-box">
                <h3>{name}</h3>
                <img src="/static/outputs/{data['img']}" alt="{name}">
                <p><span class="metric">Speed: {speed}</span> <span class="metric">Cost: {cost}</span></p>
            </div>
            '''
        else:
            html += f'''
            <div class="result-box">
                <h3>{name}</h3>
                <p><em>Manual test only</em></p>
                <p><span class="metric">Speed: ~{speed}</span> <span class="metric">Cost: {cost}</span></p>
            </div>
            '''
    
    html += '''
        </div>
        <h2>Cleaned Versions (Threshold + Median + Gaussian)</h2>
        <div class="grid">
    '''
    
    for name, data in cleanup_results.items():
        if data['img']:
            html += f'''
            <div class="result-box">
                <h3>{name} (Cleaned)</h3>
                <img src="/static/outputs/{data['img']}" alt="{name} cleaned">
            </div>
            '''
    
    html += '''
        </div>
        <br>
        <a href="/">Test Another Image</a>
    </body>
    </html>
    '''
    
    return html

if __name__ == '__main__':
    app.run(debug=True)