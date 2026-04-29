import os, io, json, base64, re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from google import genai
from google.genai import types
 
app = Flask(__name__, template_folder='templates')
CORS(app)
 
# ── API KEY: reads from environment, falls back to hardcoded value ──
API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyC0oRLfS6oP7DBYWnphiLWnhlnAsM_DcC0')
client = genai.Client(api_key=API_KEY)
 
# ── MODEL FALLBACK LIST ──
MODELS = [
       'gemini-flash-lite',
       'gemini-flash-latest',   

]
 
STORY_PROMPTS = {
    'funny':     'Write a HILARIOUS, witty, comedy-style story about this image. it should specigy the image. Use jokes, puns, and absurd humor.',
    'emotional': 'Write a deeply EMOTIONAL, heartfelt story about this image.it should specigy the image. Touch the soul, evoke nostalgia or longing.',
    'dramatic':  "Write an intensely DRAMATIC story about this image as if it's a Hollywood blockbuster scene. it should specigy the image.",
    'formal':    'Write a FORMAL, academic, professional analysis and story about this image. it should specigy the image.',
    'kid':       'Write a super FUN and SIMPLE story for KIDS! Use simple words, rhymes, make it magical!. it should specigy the image.',
    'detective': 'Write a DETECTIVE NOIR story about this image. You are a hardboiled detective narrating a mystery. it should specigy the image.',
}
 
# ── HELPERS ──
 
def open_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    # Convert palette/RGBA modes that Gemini rejects to RGB
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    return img
 
def strip_fences(raw: str) -> str:
    """Robustly extract JSON from model output that may contain markdown fences."""
    raw = raw.strip()
    # Try regex first — handles nested fences safely
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if match:
        return match.group(1).strip()
    # No fences — return as-is (model obeyed instructions)
    return raw
 
def safe_json(raw: str):
    """Parse JSON and raise a clean error on failure."""
    try:
        return json.loads(strip_fences(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {e}\n\nRaw output:\n{raw[:300]}")
 
def _should_fallback(e: Exception) -> bool:
    s = str(e)
    return any(x in s for x in [
        '429', 'RESOURCE_EXHAUSTED', 'quota',
        '404', 'NOT_FOUND', 'not found for API',
        'not supported for generateContent',
    ])
 
def generate(contents, system_instruction=None):
    """Try each model in MODELS, falling back on quota/404 errors."""
    last_err = None
    config = types.GenerateContentConfig(system_instruction=system_instruction) if system_instruction else None
    for model in MODELS:
        try:
            kwargs = dict(model=model, contents=contents)
            if config:
                kwargs['config'] = config
            return client.models.generate_content(**kwargs)
        except Exception as e:
            if _should_fallback(e):
                last_err = e
                continue
            raise  # non-quota error — surface immediately
    raise last_err  # all models exhausted
 
# ── ROUTES ──
 
@app.route('/')
def home():
    return render_template('index.html')
 
 
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
 
        image = open_image(request.files['image'].read())
        story_style = request.form.get('story_style', 'dramatic')
        story_prompt = STORY_PROMPTS.get(story_style, STORY_PROMPTS['dramatic'])
 
        prompt = f"""Analyze this image and return ONLY a valid JSON object with these exact keys.
No markdown, no code fences, no extra text — raw JSON only.
 
{{
  "caption": "one punchy sentence caption",
  "summary": "2-3 sentence summary",
  "key_objects": ["obj1","obj2","obj3","obj4","obj5"],
  "mood": "overall mood/atmosphere",
  "scene_type": "type of scene",
  "quality_score": 8,
  "keywords": ["kw1","kw2","kw3","kw4","kw5","kw6"],
  "colors": ["color1","color2","color3"],
  "story": "story text here"
}}
 
For the story field: {story_prompt}
Return ONLY the JSON object, nothing else."""
 
        response = generate([image, prompt])
        data = safe_json(response.text)
        return jsonify({'success': True, 'result': data})
 
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
 
 
@app.route('/compare', methods=['POST'])
def compare():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Two images required'}), 400
 
        img1 = open_image(request.files['image1'].read())
        img2 = open_image(request.files['image2'].read())
 
        prompt = """Compare these two images.
Return ONLY a valid JSON object — no markdown, no code fences, raw JSON only:
{
  "image1_caption": "caption for image 1",
  "image2_caption": "caption for image 2",
  "similarities": ["sim1","sim2","sim3"],
  "differences": ["diff1","diff2","diff3"],
  "mood_comparison": "how moods differ or match",
  "quality_scores": {"image1": 7, "image2": 8},
  "verdict": "which is more compelling and why",
  "combined_story": "a creative story connecting both images"
}"""
 
        response = generate([img1, img2, prompt])
        data = safe_json(response.text)
        return jsonify({'success': True, 'result': data})
 
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
 
 
@app.route('/chat', methods=['POST'])
def chat():
    try:
        req = request.get_json(force=True)
        if not req:
            return jsonify({'error': 'Invalid JSON body'}), 400
 
        question   = req.get('question', '').strip()
        image_b64  = req.get('image_b64', '')
        history    = req.get('history', [])
 
        if not question:
            return jsonify({'error': 'No question provided'}), 400
 
        content_parts = []
 
        if image_b64:
            img_bytes = base64.b64decode(image_b64)
            image = open_image(img_bytes)
            content_parts.append(image)
 
        history_text = '\n'.join(
            [f"{m['role'].upper()}: {m['text']}" for m in history[-6:]]
        )
        full_q = f"{history_text}\nUSER: {question}" if history_text else question
        content_parts.append(full_q)
 
        response = generate(
            content_parts,
            system_instruction=(
                "You are a visual AI assistant. "
                "Answer questions about the uploaded image concisely and accurately. "
                "If no image is present, say so politely."
            )
        )
        return jsonify({'success': True, 'answer': response.text.strip()})
 
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
 
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)