"""
FloodSense AI – PowerPoint Generator
Uses the CosmicSigns template and replaces text via direct XML manipulation.
Output: docs/FloodSense_AI_Presentation.pptx
"""
import shutil, sys, os, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor
from lxml import etree

# ── paths ──────────────────────────────────────────────────────────────────────
TEMPLATE_PATH = (
    "C:/Users/achal/.gemini/antigravity/brain/"
    "02a94fc8-b709-4c4f-a142-030c5bb9e863/.user_uploaded/"
    "media_1793501088823_5a3608e9.pptx"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(SCRIPT_DIR, "FloodSense_AI_Presentation.pptx")

shutil.copy(TEMPLATE_PATH, OUT_PATH)
prs = Presentation(OUT_PATH)

# Namespace
NSMAP = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}

def get_all_text_elements(slide):
    """Return list of (element, full_text) for all <a:t> parents (<a:r>)."""
    return slide._element.findall('.//a:t', NSMAP)

def replace_text_in_slide(slide, old_text, new_text):
    """
    Find the <a:t> element(s) whose combined paragraph text equals old_text,
    and replace with new_text (may contain \\n for line breaks via <a:br>).
    """
    # We work at the <a:p> paragraph level
    paras = slide._element.findall('.//a:p', NSMAP)
    replaced = 0
    for para in paras:
        runs = para.findall('.//a:r', NSMAP)
        full = ''.join(r.findtext('a:t', default='', namespaces=NSMAP) for r in runs).strip()
        if full == old_text or full.replace('\xa0', ' ') == old_text:
            if not new_text:  # blank it out
                for r in runs:
                    t_el = r.find('a:t', NSMAP)
                    if t_el is not None:
                        t_el.text = ''
                replaced += 1
                continue
            # Replace first run's text, remove the rest
            lines = new_text.split('\n')
            if runs:
                first_run = runs[0]
                t_el = first_run.find('a:t', NSMAP)
                if t_el is not None:
                    t_el.text = lines[0]
                # remove extra runs
                for r in runs[1:]:
                    para.remove(r)
                # add <a:br> + run for additional lines
                for extra_line in lines[1:]:
                    # clone the first run for formatting
                    import copy
                    br = etree.SubElement(para, '{http://schemas.openxmlformats.org/drawingml/2006/main}br')
                    # optionally copy rPr
                    rPr = first_run.find('a:rPr', NSMAP)
                    if rPr is not None:
                        br_rPr = copy.deepcopy(rPr)
                        br.insert(0, br_rPr)
                    new_run = copy.deepcopy(first_run)
                    new_run.find('a:t', NSMAP).text = extra_line
                    para.append(new_run)
            replaced += 1
    return replaced


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-slide replacement maps  {old_text: new_text}
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SLIDE_MAPS = [

    # ── Slide 1: Cover ─────────────────────────────────────────────────────────
    {
        'Real-Time ASL to Text':
            'AI Manthan 2.0  |  Problem Statement 11',
        'CosmicSigns':
            'FloodSense AI',
        'Signed. Heard. Connected.':
            'Predict Before It Pools',
        'Team Cosmos':
            'Team Cosmos',
        'Achal Goyal, Deepanshi Mane,':
            'Achal Goyal  |  Deepanshi Mane  |',
        'Vanshvi':
            'Vanshvi Jain',
        'Jain':
            '',
        'Department of Information Technology':
            'Department of Information Technology',
        'Acropolis Institute of Technology and Research, Indore':
            'Acropolis Institute of Technology and Research, Indore',
    },

    # ── Slide 2: Problem & Solution ────────────────────────────────────────────
    {
        'IDEA TITLE: COSMICSIGNS':
            'PROBLEM STATEMENT 11  |  FLOODSENSE AI',
        'Proposed Solution':
            'Problem & Proposed Solution',
        'Tablet system that turns ASL into text in real time':
            "Mumbai's Dadar-Hindmata-Parel corridor floods every monsoon.",
        'Two-way chat: signer and staff talk by text, voice or tap':
            'Hindmata junction sits in a saucer depression — when Arabian Sea',
        'How it addresses the problem':
            'tides exceed 3.51 m CD, drains reverse and pumps are the only relief.',
        'No interpreter needed: lower cost, no waiting':
            'Current response is REACTIVE. Pumps dispatched only after flooding.',
        'Text shows on both screens. No notes, no lip-reading':
            'FloodSense AI brings proactive, data-driven flood management:',
        'Fits hospitals, banks, government desks and shops':
            '',
        'Innovation and uniqueness':
            'Our Solution',
        'Letter and word models switch automatically':
            '• ST-GNN model predicts flood depth at 110 junctions (1–3 hrs ahead)',
        'AI suggests replies before the sign is even finished':
            '• 70.5% flood onset recall vs 0.0% for the persistence baseline',
        'Works offline. No login and no data stored':
            '• Pump optimizer dispatches proactively, reducing flood exposure 9.4%',
        '99.8%':
            '9.4%',
        'letter model':
            'flood exposure cut',
        'F-score':
            'vs current practice',
        '100%':
            '70.5%',
        'core signing':
            'flood onset recall',
        'works offline':
            'vs 0.0% baseline',
    },

    # ── Slide 3: Technical Approach ────────────────────────────────────────────
    {
        'TECHNICAL APPROACH':
            'TECHNICAL APPROACH',
        'Technologies Used':
            'Technologies Used',
        'On-device AI':
            'ST-GNN Model',
        'MediaPipe 21-point hand tracking and TFLite ASL models':
            'PyTorch Geometric · Spatio-Temporal GNN\nRecall 91.5% | MAE 1.18 cm | 110 nodes',
        'Cloud AI (online mode)':
            'FastAPI Backend',
        'Gemini via OpenRouter, Google Translate, Speech-to-Text':
            'REST API serving predictions, pump commands\nand bilingual flood alerts in real time',
        'App':
            'Streamlit Dashboard',
        'Tablet app with camera, chat, keyboard and text-to-speech':
            'Interactive flood map, timeline slider,\npump status panel and color-coded alerts',
        'Methodology and Process':
            'System Pipeline',
        'Camera':
            'Rain + Tide',
        'captures the signer\'s hands':
            'real-time sensor inputs',
        'MediaPipe':
            'ST-GNN',
        'tracks 21 hand points':
            'depth forecast 1-3 hrs',
        'Smart routing':
            'Pump Optimizer',
        'pause = letter, motion = word':
            'rule-based dispatch',
        'TFLite models':
            'Alert Engine',
        'read letters and word signs':
            'bilingual notification',
        'Chat + AI':
            'Dashboard',
        'text, suggestions, voice':
            'citizen-facing UI',
        'Filters stop false signs: EMA smoothing, 4-frame check, 10% confidence margin':
            'Trained on real OpenStreetMap drain topology + Copernicus GLO-30 DEM elevation data',
        'Core ASL works offline. AI suggestions and Google Translate need internet':
            'Validated on 46 unseen test storms across Light (40mm), Heavy (120mm) and Extreme (250mm) scenarios',
    },

    # ── Slide 4: Feasibility & Challenges ──────────────────────────────────────
    {
        'FEASIBILITY AND VIABILITY':
            'FEASIBILITY, CHALLENGES & SOLUTIONS',
        'Feasibility':
            'Feasibility',
        'Runs on a normal tablet or laptop camera':
            'Real OSM drain graph + Copernicus DEM elevation',
        'AI models run on the device, no heavy server':
            'Runs in browser via Streamlit — no GPU required',
        'Trained on our own collected ASL data':
            'Validated on 46 unseen storms with dual baseline',
        'Working prototype already tested live':
            'Open-source stack: PyTorch, FastAPI, Streamlit',
        'Challenges &amp; Risks':
            'Challenges',
        'Some signs look alike and can be misread':
            'Sparse labeled monsoon sensor data',
        'Only 8 word signs so far':
            'Tide-flood interaction nonlinearity',
        'Light, camera angle and signer style change results':
            'Class imbalance (flood events are rare)',
        'AI suggestions and translation need internet':
            'Multi-junction spatial dependency',
        'How We Overcome':
            'How We Overcome',
        'EMA smoothing, 4-frame check and 10% confidence margin':
            '12x loss penalty for missed flood events',
        'Position-invariant keypoints, not raw pixels':
            'Synthetic storm scenarios (40 / 120 / 250 mm)',
        'Grow to 100+ signs and add personal signer models':
            'Dual baseline: Persistence + Passive drainage',
        'Offline mode with preset suggestions and Hindi dictionary':
            'Graph convolution captures spatial drain topology',
    },

    # ── Slide 5: Impact & Benefits ─────────────────────────────────────────────
    {
        'IMPACT AND BENEFITS':
            'IMPACT AND BENEFITS',
        'Impact on the Target Audience':
            'Impact on Mumbai & Beyond',
        'Deaf and hard-of-hearing people can talk at any counter on their own':
            'FloodSense AI reduces flood exposure 9.4% vs current municipal practice',
        'Faster, clearer service. No writing notes':
            'ST-GNN detects flood onset at Hindmata 1+ hr ahead with 70.5% recall',
        'Staff just speak or tap. Text appears instantly':
            'Bilingual alerts (English + Hindi) reach citizens before inundation',
        'Works in hospitals, banks, government offices, hotels and shops':
            'Scalable: tested on Mumbai Dadar-Hindmata (N=110) & Kurla-Mithi (N=91)',
        '0':
            '91.5%',
        'personal data stored':
            'Flood Detection\nRecall (ST-GNN)',
        'Hindi first':
            '70.5%',
        'plus dozens more online':
            'Onset Recall at +1h\nvs 0.0% baseline',
        'Benefits':
            'Benefits',
        'Social':
            'Social',
        'Equal access, independence and dignity for deaf people':
            'Early warning enables timely evacuation and saves lives',
        'Economic':
            'Economic',
        'No interpreter at every counter. One tablet per counter':
            'Proactive pump dispatch cuts infrastructure damage cost',
        'Environmental':
            'Environmental',
        'Paperless talks. Uses devices that already exist':
            'Reduces urban waterlogging and associated health risks',
    },

    # ── Slide 6: Research & References ─────────────────────────────────────────
    {
        'RESEARCH AND REFERENCES':
            'DATA SOURCES & REFERENCES',
        'Research and Tools':
            'Real Data Sources',
        'Google MediaPipe Hands: 21-point hand tracking':
            'OpenStreetMap (osmnx) — Mumbai drain network topology',
        'developers.google.com/mediapipe':
            'Copernicus GLO-30 DEM — 30 m elevation data',
        'TensorFlow Lite: on-device model inference':
            'BRIMSTOWAD — pump capacities (3–6 m3/s per station)',
        'tensorflow.org/lite':
            'IMD rainfall records — monsoon intensity data',
        'Gemini via OpenRouter: smart reply suggestions':
            'PyTorch Geometric — ST-GNN framework',
        'openrouter.ai':
            'Streamlit + FastAPI + Folium — dashboard & API stack',
        'Google Cloud Translation and Speech-to-Text':
            'Key References',
        'cloud.google.com/translate':
            'Yao et al. (2018) DCRNN | Shi et al. (2015) ConvLSTM',
        'Our own ASL data collection and model training (5-fold CV)':
            'AI Manthan 2.0 PS-11 — Urban Flood Management\nGitHub: Team Cosmos · FloodSense AI',
        'Working prototype: signing and chatting live with AI suggestions':
            'Live Streamlit dashboard — flood map, timeline, pump status & bilingual alerts',
    },

    # ── Slide 7: Closing ────────────────────────────────────────────────────────
    {
        'Thank You':
            'Thank You',
        'Signed. Heard. Connected.':
            'Predict. Prevent. Protect.',
        'Breaking the silence – one counter at a time.':
            "Turning Mumbai's monsoon crisis into a manageable forecast\n— one junction at a time.",
        'Cosmos':
            'Cosmos',
    },
]

# ── Apply all replacements ──────────────────────────────────────────────────────
print("\n=== Generating FloodSense AI Presentation ===\n")

for si, slide_map in enumerate(SLIDE_MAPS):
    slide = prs.slides[si]
    print(f"\n--- Slide {si+1} ---")
    for old, new in slide_map.items():
        n = replace_text_in_slide(slide, old, new)
        if n:
            print(f"  OK  {old[:50]!r}")
        else:
            print(f"  --  NOT FOUND: {old[:50]!r}")

prs.save(OUT_PATH)
print(f"\n✅ Saved: {OUT_PATH}")
print(f"   Total slides: {len(prs.slides)}")
