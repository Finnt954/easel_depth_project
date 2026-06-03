from flask import Flask, request, send_file
from PIL import Image
import time
import os
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import matplotlib.pyplot as plt

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
            input, select, button {
                padding: 10px 20px;
                margin: 10px;
                border-radius: 6px;
                border: none;
                font-size: 16px;
            }
            select { background: #0f3460; color: white; }
            button {
                background: #e94560;
                color: white;
                cursor: pointer;
                font-weight: bold;
            }
            button:hover { background: #c23152; }
            .result-box {
                background: #16213e;
                padding: 20px;
                margin: 20px;
                border-radius: 12px;
                display: inline-block;
                vertical-align: top;
            }
            .grid { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
            a { color: #e94560; }
        </style>
    </head>
    <body>
        <h1>Depth Model Tester</h1>
        <form action="/run" method="post" enctype="multipart/form-data">
            <input type="file" name="image" required><br>
            <button type="submit" name="mode" value="all">Compare All Models</button>
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
    
    # Gemini placeholder
    results['Gemini'] = {'time': 3.5, 'cost': '~$0.00002 (API)', 'img': None}
    
    # Build comparison HTML
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
            h1 { color: #e94560; }
            .grid { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
            .result-box {
                background: #16213e;
                padding: 20px;
                border-radius: 12px;
                width: 300px;
            }
            .result-box img { width: 100%; border-radius: 8px; }
            .metric { color: #0f3460; background: #eee; padding: 4px 8px; border-radius: 4px; display: inline-block; margin: 4px; }
            a { color: #e94560; }
            .winner { border: 2px solid #e94560; }
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
        <br>
        <a href="/">Test Another Image</a>
    </body>
    </html>
    '''
    
    return html

if __name__ == '__main__':
    app.run(debug=True)