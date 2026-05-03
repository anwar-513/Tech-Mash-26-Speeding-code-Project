import os, io, json, base64, re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from google import genai
from google.genai import types

app = Flask(__name__, template_folder='templates')
CORS(app)

# ── API KEY ──
API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyAV_2cB9hP6xNT6MRRpOJGpOzDt2Pggcts')
client = genai.Client(api_key=API_KEY)

# ── MODEL FALLBACK LIST ──
MODELS = [
    'gemini-flash-lite-latest'
]

STORY_PROMPTS = {
    'funny':     'Write a HILARIOUS, witty, comedy-style story specifically about what you see in this image. Use jokes, puns, and absurd humor. Reference actual objects and scenes visible.',
    'emotional': 'Write a deeply EMOTIONAL, heartfelt story specifically about what you see in this image. Reference actual elements visible. Touch the soul, evoke nostalgia or longing.',
    'dramatic':  "Write an intensely DRAMATIC story about this exact image as if it's a Hollywood blockbuster scene. Reference what you actually see.",
    'formal':    'Write a FORMAL, academic, professional analysis and narrative about this specific image. Reference what you actually observe.',
    'kid':       'Write a super FUN and SIMPLE story for KIDS about exactly what you see in this image! Use simple words, rhymes, make it magical!',
    'detective': 'Write a DETECTIVE NOIR story about this exact image. You are a hardboiled detective narrating what you see as clues to a mystery.',
}

# ── HELPERS ──

def open_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    return img

def strip_fences(raw: str) -> str:
    raw = raw.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if match:
        return match.group(1).strip()
    return raw

def safe_json(raw: str):
    try:
        return json.loads(strip_fences(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON. Raw output:\n{raw[:400]}")

def _should_fallback(e: Exception) -> bool:
    s = str(e)
    return any(x in s for x in [
        '429', 'RESOURCE_EXHAUSTED', 'quota',
        '404', 'NOT_FOUND', 'not found for API',
        'not supported for generateContent',
    ])

def generate(contents, system_instruction=None):
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
            raise
    raise last_err

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

        prompt = f"""Analyze this image carefully and return ONLY a valid JSON object.
No markdown, no code fences, no extra text — raw JSON only.

{{
  "caption": "one punchy sentence caption describing THIS specific image",
  "summary": "2-3 sentence summary of exactly what is in this image",
  "key_objects": ["specific object 1","specific object 2","specific object 3","specific object 4","specific object 5"],
  "mood": "the overall mood/atmosphere of this image",
  "scene_type": "exact type of scene (e.g. outdoor portrait, urban street, food photography)",
  "quality_score": 8,
  "keywords": ["keyword1","keyword2","keyword3","keyword4","keyword5","keyword6"],
  "colors": ["dominant color 1","dominant color 2","dominant color 3"],
  "story": "story text here — must reference actual elements from THIS image"
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

        prompt = """Compare these two images carefully.
Return ONLY a valid JSON object — no markdown, no code fences, raw JSON only:
{
  "image1_caption": "specific caption for the FIRST image",
  "image2_caption": "specific caption for the SECOND image",
  "similarities": ["specific similarity 1","specific similarity 2","specific similarity 3"],
  "differences": ["specific difference 1","specific difference 2","specific difference 3"],
  "mood_comparison": "how the moods of the two images differ or match",
  "quality_scores": {"image1": 7, "image2": 8},
  "verdict": "which image is more compelling and exactly why",
  "combined_story": "a creative story that ties both images together"
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

        question  = req.get('question', '').strip()
        image_b64 = req.get('image_b64', '')
        history   = req.get('history', [])

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
                "Always reference specific things you can see in the image. "
                "If no image is present, say so politely."
            )
        )
        return jsonify({'success': True, 'answer': response.text.strip()})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
