import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel, AutoModelForCausalLM
from tqdm import tqdm
import json
import http.server
import socketserver
from http import HTTPStatus
import numpy as np
import socket
from pathlib import Path
import base64
import os
import sqlite3
from datetime import date

# --- START: Import modules for decryption ---
try:
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    # Add AESGCM for symmetric decryption
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:
    print("\n❌ Missing dependency: 'cryptography'. Please install it by running:")
    print("   pip install cryptography")
    exit(1)
# --- END: Import modules for decryption ---

# --- START: Import modules for Google Auth ---
try:
    from google.oauth2 import id_token
    from google.auth.transport import requests
except ImportError:
    print("\n❌ Missing dependency: 'google-auth'. Please install it by running:")
    print("   pip install google-auth")
    exit(1)
# --- END: Import modules for Google Auth ---

# --- NLTK and other imports are unchanged ---
import nltk
def check_and_download_nltk_data():
    required_packages = ['punkt', 'averaged_perceptron_tagger']
    print("Checking for required NLTK data...")
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}.zip')
        except LookupError:
            nltk.download(package, quiet=True)
    print("NLTK data check complete.")
check_and_download_nltk_data()

import textstat
from nltk.tokenize import sent_tokenize, word_tokenize

# --- Device Setup is unchanged ---
def get_optimal_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
DEVICE = get_optimal_device()

# ==============================================================================
#  AUTHENTICATION & QUOTA SETUP
# ==============================================================================
# IMPORTANT: Replace with your actual Google Client ID from the Google Cloud Console
GOOGLE_CLIENT_ID = "YOUR_CLIENT_ID.apps.googleusercontent.com" 
DB_PATH = "users.db"
DAILY_WORD_LIMIT = 1000
PEMBROKE_DOMAIN = "@pembroke.sa.edu.au"

def init_db():
    """Initializes the database and creates the users table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            google_id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            words_used_today INTEGER DEFAULT 0,
            last_reset_date TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully.")

def get_db_connection():
    """Returns a new database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ==============================================================================
#  ENCRYPTION/DECRYPTION SETUP
# ==============================================================================
PRIVATE_KEY = None

def load_private_key():
    global PRIVATE_KEY
    key_path = Path("private_key.pem")
    if not key_path.exists():
        print("\n❌ CRITICAL ERROR: 'private_key.pem' not found.")
        exit(1)
    
    with open(key_path, "rb") as key_file:
        try:
            PRIVATE_KEY = serialization.load_pem_private_key(
                key_file.read(),
                password=None,
            )
            print("✅ Private key loaded successfully for request decryption.")
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: Failed to load 'private_key.pem'. Error: {e}")
            exit(1)

def decrypt_payload(payload):
    """
    Decrypts a hybrid encryption payload.
    Returns the decrypted text AND the AES key used.
    """
    if not PRIVATE_KEY:
        raise ConnectionAbortedError("Server private key is not loaded.")

    # --- Step 1: Decrypt the AES key with RSA ---
    encrypted_aes_key_b64 = payload['encrypted_key']
    encrypted_aes_key_bytes = base64.b64decode(encrypted_aes_key_b64)

    decrypted_b64_aes_key_bytes = PRIVATE_KEY.decrypt(
        encrypted_aes_key_bytes,
        padding.PKCS1v15()
    )
    
    aes_key_bytes = base64.b64decode(decrypted_b64_aes_key_bytes)

    # --- Step 2: Decrypt the text with AES-GCM ---
    iv_b64 = payload['iv']
    iv_bytes = base64.b64decode(iv_b64)
    
    encrypted_text_b64 = payload['encrypted_text']
    encrypted_text_bytes = base64.b64decode(encrypted_text_b64)

    aesgcm = AESGCM(aes_key_bytes)
    decrypted_text_bytes = aesgcm.decrypt(iv_bytes, encrypted_text_bytes, None)
    
    # Return both the text and the key for reuse
    return decrypted_text_bytes.decode('utf-8'), aes_key_bytes


# --- All model classes (LinguisticAnalyzer, LogitDetector, etc.) are unchanged ---
class LinguisticAnalyzer:
    def __init__(self, perplexity_model, perplexity_tokenizer, device):
        self.perplexity_model = perplexity_model
        self.perplexity_tokenizer = perplexity_tokenizer
        self.device = device
        print("LinguisticAnalyzer initialized.")
    def analyze(self, text):
        if len(text.split()) < 10: return None
        readability_grade = textstat.flesch_kincaid_grade(text)
        sentences = sent_tokenize(text)
        if len(sentences) < 2: return { "readability_grade": round(readability_grade, 1), "sentence_length_variation": 0, "lexical_richness": 0, "perplexity": 0 }
        sentence_lengths = [len(word_tokenize(s)) for s in sentences]
        len_std_dev = np.std(sentence_lengths)
        words = word_tokenize(text.lower())
        lexical_richness = len(set(words)) / len(words) if words else 0
        try:
            encodings = self.perplexity_tokenizer(text, return_tensors='pt')
            input_ids = encodings.input_ids.to(self.device)
            if input_ids.shape[1] > 1024: input_ids = input_ids[:, :1024]
            with torch.no_grad():
                outputs = self.perplexity_model(input_ids, labels=input_ids)
                perplexity = torch.exp(outputs.loss).item()
        except Exception: perplexity = 0
        return { "readability_grade": round(readability_grade, 1), "sentence_length_variation": round(len_std_dev, 2), "lexical_richness": round(lexical_richness, 2), "perplexity": round(perplexity, 1) }

class LogitDetector:
    def __init__(self, model_name, device=DEVICE):
        self.device, self.model, self.tokenizer = device, AutoModelForCausalLM.from_pretrained(model_name).to(device), AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self, config):
        super().__init__(config); self.model, self.classifier = AutoModel.from_config(config), nn.Linear(config.hidden_size, 1); self.init_weights()
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask); last_hidden_state = outputs[0]; input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1); sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask; logits = self.classifier(pooled_output); return {"logits": logits}

class ClassifierDetector:
    def __init__(self, model_dir, device=DEVICE):
        self.device, self.tokenizer, self.model = device, AutoTokenizer.from_pretrained(model_dir), DesklibAIDetectionModel.from_pretrained(model_dir).to(device); self.model.eval()
    def detect(self, text, max_len=512):
        encoded = self.tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.device); attention_mask = encoded['attention_mask'].to(self.device)
        with torch.no_grad(): outputs = self.model(input_ids=input_ids, attention_mask=attention_mask); probability = torch.sigmoid(outputs["logits"]).item()
        return probability

class DesklibHighlighter:
    def __init__(self, classifier_detector): self.classifier = classifier_detector
    def highlight(self, text, threshold=0.65):
        sentences = sent_tokenize(text); chunks = []
        if not sentences: return []
        for sentence in tqdm(sentences, desc="Highlighting Analysis", leave=False):
            if not sentence.strip(): continue
            prob = self.classifier.detect(sentence); sentence_type = 'AI' if prob > threshold else 'Human'; chunks.append({"text": sentence, "type": sentence_type})
        return chunks

class CombinedDetector:
    def __init__(self):
        self.logit_detector = LogitDetector(model_name="./models/distilgpt2")
        self.classifier_detector = ClassifierDetector(model_dir="./models/desklib-detector")
        self.desklib_highlighter = DesklibHighlighter(self.classifier_detector)
        self.linguistic_analyzer = LinguisticAnalyzer(perplexity_model=self.logit_detector.model, perplexity_tokenizer=self.logit_detector.tokenizer, device=DEVICE)
    def _calculate_ensemble_score(self, desklib_prob, highlight_chunks, linguistic_stats):
        score_desklib = desklib_prob
        ai_char_count = sum(len(c['text']) for c in highlight_chunks if c['type'] == 'AI')
        total_char_count = sum(len(c['text']) for c in highlight_chunks)
        score_highlighting = (ai_char_count / total_char_count) if total_char_count > 0 else 0.0
        score_linguistic = 0.0
        if linguistic_stats:
            perplexity = linguistic_stats.get('perplexity', 100)
            perplexity_score = 1.0 - ((perplexity - 40) / (100 - 40))
            perplexity_score = np.clip(perplexity_score, 0, 1)
            sent_variation = linguistic_stats.get('sentence_length_variation', 8)
            burstiness_score = 1.0 - ((sent_variation - 2) / (8 - 2))
            burstiness_score = np.clip(burstiness_score, 0, 1)
            score_linguistic = (perplexity_score + burstiness_score) / 2.0
        W_DESKLIB, W_HIGHLIGHT, W_LINGUISTIC = 0.60, 0.25, 0.15
        final_score = (W_DESKLIB * score_desklib) + (W_HIGHLIGHT * score_highlighting) + (W_LINGUISTIC * score_linguistic)
        component_scores = {"desklib_overall_score": round(score_desklib, 4), "desklib_highlighting_score": round(score_highlighting, 4), "linguistic_feature_score": round(score_linguistic, 4), "weights": {"desklib_overall": W_DESKLIB, "desklib_highlighting": W_HIGHLIGHT, "linguistic": W_LINGUISTIC}}
        return final_score, component_scores
    def detect(self, text):
        print("\n--- Starting New Analysis ---")
        desklib_prob = self.classifier_detector.detect(text, max_len=768)
        highlight_chunks = self.desklib_highlighter.highlight(text)
        linguistic_stats = self.linguistic_analyzer.analyze(text)
        final_prob, component_scores = self._calculate_ensemble_score(desklib_prob=desklib_prob, highlight_chunks=highlight_chunks, linguistic_stats=linguistic_stats)
        result = { "overall_percentage": round(final_prob * 100, 2), "component_scores": component_scores, "chunks": highlight_chunks, "linguistics": linguistic_stats }
        return result

# ==============================================================================
#  SERVER SETUP
# ==============================================================================
print("Creating CombinedDetector instance...")
DETECTOR_INSTANCE = CombinedDetector()
print("\n✅ All models loaded. Server is ready.")

class AIRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(HTTPStatus.OK)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == '/detect':
            decrypted_text = None
            aes_key_bytes = None
            conn = None
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                body = json.loads(post_data)

                # --- 1. AUTHENTICATION ---
                id_token_str = body.get('token')
                if not id_token_str:
                    raise ValueError("Request is missing authentication token.")
                
                print("Request received. Verifying token...")
                try:
                    idinfo = id_token.verify_oauth2_token(id_token_str, requests.Request(), GOOGLE_CLIENT_ID)
                    google_id = idinfo['sub']
                    email = idinfo['email']
                    print(f"✅ Token verified for user: {email}")
                except ValueError as e:
                    # Handle invalid token
                    print(f"❌ Token verification failed: {e}")
                    self.send_response(HTTPStatus.UNAUTHORIZED)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Invalid authentication token.'}).encode('utf-8'))
                    return
                
                # --- 2. DECRYPTION ---
                if 'encrypted_key' not in body or 'iv' not in body or 'encrypted_text' not in body:
                    raise ValueError("Request is missing required encryption fields.")
                
                print("Decrypting payload...")
                decrypted_text, aes_key_bytes = decrypt_payload(body)
                word_count = len(decrypted_text.split())

                # --- 3. QUOTA CHECK ---
                print(f"Word count for this request: {word_count}")
                conn = get_db_connection()
                cursor = conn.cursor()
                today_str = date.today().isoformat()
                
                cursor.execute("SELECT * FROM users WHERE google_id = ?", (google_id,))
                user = cursor.fetchone()

                is_unlimited = email.endswith(PEMBROKE_DOMAIN)
                words_used_today = 0
                
                if user:
                    if user['last_reset_date'] != today_str:
                        print(f"New day for user {email}. Resetting quota.")
                        cursor.execute("UPDATE users SET words_used_today = 0, last_reset_date = ? WHERE google_id = ?", (today_str, google_id))
                        words_used_today = 0
                    else:
                        words_used_today = user['words_used_today']
                else:
                    print(f"New user: {email}. Creating database entry.")
                    cursor.execute("INSERT INTO users (google_id, email, last_reset_date) VALUES (?, ?, ?)", (google_id, email, today_str))

                conn.commit()

                if not is_unlimited:
                    if words_used_today + word_count > DAILY_WORD_LIMIT:
                        print(f"❌ Quota exceeded for user {email}.")
                        self.send_response(HTTPStatus.TOO_MANY_REQUESTS)
                        self.send_header('Content-type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        error_payload = {
                            'error': 'Daily word limit exceeded.',
                            'quota_info': {
                                'words_used_today': words_used_today,
                                'daily_limit': DAILY_WORD_LIMIT,
                                'is_unlimited': is_unlimited
                            }
                        }
                        self.wfile.write(json.dumps(error_payload).encode('utf-8'))
                        return

                # --- 4. INFERENCE ---
                print("Quota check passed. Running inference...")
                results = DETECTOR_INSTANCE.detect(decrypted_text)

                # --- 5. UPDATE QUOTA & PREPARE RESPONSE ---
                if not is_unlimited:
                    new_total_words = words_used_today + word_count
                    cursor.execute("UPDATE users SET words_used_today = ? WHERE google_id = ?", (new_total_words, google_id))
                    conn.commit()
                    words_used_today = new_total_words
                
                results['quota_info'] = {
                    'words_used_today': words_used_today,
                    'daily_limit': DAILY_WORD_LIMIT,
                    'is_unlimited': is_unlimited
                }
                
                # --- 6. ENCRYPT AND SEND RESPONSE ---
                print("Inference complete. Encrypting response...")
                results_json_bytes = json.dumps(results).encode('utf-8')
                iv_response = os.urandom(12) 
                aesgcm = AESGCM(aes_key_bytes)
                encrypted_response_bytes = aesgcm.encrypt(iv_response, results_json_bytes, None)
                response_payload = {
                    'iv': base64.b64encode(iv_response).decode('utf-8'),
                    'encrypted_response': base64.b64encode(encrypted_response_bytes).decode('utf-8')
                }
                
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode('utf-8'))
                print("Encrypted response sent.")

            except Exception as e:
                print(f"Error processing request: {e}")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                error_payload = {'error': f"An error occurred on the server: {type(e).__name__}"}
                self.wfile.write(json.dumps(error_payload).encode('utf-8'))
            finally:
                if conn:
                    conn.close()
                if decrypted_text is not None:
                    print("Clearing decrypted text from memory.")
                    decrypted_text = None
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

# --- DualStackServer and run_server are unchanged ---
class DualStackServer(socketserver.TCPServer):
    allow_reuse_address = True
    address_family = socket.AF_INET6

def run_server(port=8000):
    with DualStackServer(("", port), AIRequestHandler) as httpd:
        print(f"✅ Serving at http://[::]:{port} (IPv6 + IPv4)")
        print("Point your browser to http://localhost:8000/index.html")
        httpd.serve_forever()

if __name__ == "__main__":
    init_db()
    load_private_key()
    run_server()
