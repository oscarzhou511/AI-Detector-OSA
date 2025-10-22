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

# --- START: Import modules for decryption ---
try:
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
except ImportError:
    print("\n❌ Missing dependency: 'cryptography'. Please install it by running:")
    print("   pip install cryptography")
    exit(1)
# --- END: Import modules for decryption ---


# --- Self-healing NLTK Data Check ---
import nltk
def check_and_download_nltk_data():
    required_packages = ['punkt', 'averaged_perceptron_tagger']
    print("Checking for required NLTK data...")
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}.zip')
            print(f"  - NLTK package '{package}' already installed.")
        except LookupError:
            print(f"  - NLTK package '{package}' not found. Downloading...")
            nltk.download(package, quiet=True)
            print(f"  - '{package}' downloaded successfully.")
    print("NLTK data check complete.")
check_and_download_nltk_data()

import textstat
from nltk.tokenize import sent_tokenize, word_tokenize

# --- Device Setup ---
def get_optimal_device():
    if torch.backends.mps.is_available():
        print("Metal Performance Shaders (MPS) is available. Using GPU on Mac.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    else:
        print("No GPU acceleration available. Using CPU.")
        return torch.device("cpu")
DEVICE = get_optimal_device()

# ==============================================================================
#  ENCRYPTION/DECRYPTION SETUP
# ==============================================================================
PRIVATE_KEY = None

def load_private_key():
    """Loads the RSA private key from 'private_key.pem' into a global variable."""
    global PRIVATE_KEY
    key_path = Path("private_key.pem")
    if not key_path.exists():
        print("\n❌ CRITICAL ERROR: 'private_key.pem' not found in the current directory.")
        print("Please generate an RSA key pair and place the private key file here.")
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
            print("Ensure the file is a valid, unencrypted PEM-formatted RSA private key.")
            exit(1)

def decrypt_text(encrypted_base64_text):
    """Decrypts a base64 encoded, RSA-encrypted string."""
    if not PRIVATE_KEY:
        raise ConnectionAbortedError("Server private key is not loaded. Cannot decrypt.")
    
    encrypted_bytes = base64.b64decode(encrypted_base64_text)
    
    # Client-side JSEncrypt encrypts data in chunks of key_size/8 bytes.
    # We must decrypt chunk by chunk and then join the results.
    key_size_bytes = PRIVATE_KEY.key_size // 8
    decrypted_chunks = []
    
    for i in range(0, len(encrypted_bytes), key_size_bytes):
        chunk = encrypted_bytes[i:i + key_size_bytes]
        decrypted_chunk = PRIVATE_KEY.decrypt(
            chunk,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        decrypted_chunks.append(decrypted_chunk)
        
    return b''.join(decrypted_chunks).decode('utf-8')

# ==============================================================================
#  ADVANCED FEATURE: Linguistic Analyzer (Unchanged)
# ==============================================================================
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

# ==============================================================================
#  DETECTOR 1: Logit-based Model (Unchanged)
# ==============================================================================
class LogitDetector:
    def __init__(self, model_name, device=DEVICE):
        print(f"Initializing LogitDetector with '{model_name}' (for perplexity calculation)...")
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        print("LogitDetector initialized.")

# ==============================================================================
#  DETECTOR 2: Classifier-based Detection (Unchanged)
# ==============================================================================
class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        logits = self.classifier(pooled_output)
        return {"logits": logits}

class ClassifierDetector:
    def __init__(self, model_dir, device=DEVICE):
        print(f"Initializing ClassifierDetector with '{model_dir}'...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = DesklibAIDetectionModel.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        print("ClassifierDetector initialized.")
    def detect(self, text, max_len=512):
        encoded = self.tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probability = torch.sigmoid(outputs["logits"]).item()
        return probability

# ==============================================================================
#  Desklib-based Highlighting System (Unchanged)
# ==============================================================================
class DesklibHighlighter:
    def __init__(self, classifier_detector):
        self.classifier = classifier_detector
        print("DesklibHighlighter initialized.")
    def highlight(self, text, threshold=0.65):
        sentences = sent_tokenize(text)
        if not sentences: return []
        chunks = []
        for sentence in tqdm(sentences, desc="Highlighting Analysis", leave=False):
            if not sentence.strip(): continue
            prob = self.classifier.detect(sentence)
            sentence_type = 'AI' if prob > threshold else 'Human'
            chunks.append({"text": sentence, "type": sentence_type})
        return chunks

# ==============================================================================
#  Combined Detector (Unchanged)
# ==============================================================================
class CombinedDetector:
    def __init__(self):
        logit_model_path = "./models/distilgpt2"
        classifier_model_path = "./models/desklib-detector"
        print(f"Loading LogitDetector from local path: {logit_model_path} (for perplexity)")
        self.logit_detector = LogitDetector(model_name=logit_model_path)
        print(f"Loading ClassifierDetector from local path: {classifier_model_path}")
        self.classifier_detector = ClassifierDetector(model_dir=classifier_model_path)
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
print("\n✅ All models loaded. Server is ready to accept requests.")

class AIRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(HTTPStatus.OK)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == '/detect':
            decrypted_text = None  # Ensure variable exists for the 'finally' block
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                body = json.loads(post_data)
                
                # --- START: DECRYPTION AND SECURE HANDLING ---
                encrypted_text = body.get('encrypted_text')
                if not encrypted_text or not isinstance(encrypted_text, str):
                    raise ValueError("Request body must contain non-empty 'encrypted_text' field")
                
                # 1. Just-in-Time Decryption: Decrypt only when ready to process.
                print("Request received. Decrypting payload...")
                decrypted_text = decrypt_text(encrypted_text)
                
                # 2. Run Inference: Pass the plaintext to the detector.
                print("Payload decrypted. Running inference...")
                results = DETECTOR_INSTANCE.detect(decrypted_text)
                # --- END: DECRYPTION AND SECURE HANDLING ---
                
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(results).encode('utf-8'))
                print("Analysis complete and results sent.")

            except Exception as e:
                print(f"Error processing request: {e}")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                error_payload = {'error': f"An error occurred on the server: {type(e).__name__}"}
                self.wfile.write(json.dumps(error_payload).encode('utf-8'))
            finally:
                # 3. Secure Deletion: Overwrite the variable holding the plaintext
                #    to ensure it's cleared from memory as soon as possible.
                if decrypted_text is not None:
                    print("Clearing decrypted text from memory.")
                    decrypted_text = None
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()


class DualStackServer(socketserver.TCPServer):
    allow_reuse_address = True
    address_family = socket.AF_INET6

def run_server(port=8000):
    with DualStackServer(("", port), AIRequestHandler) as httpd:
        print(f"✅ Serving at http://[::]:{port} (IPv6 + IPv4)")
        print("Point your browser to http://localhost:8000/index.html")
        httpd.serve_forever()

if __name__ == "__main__":
    load_private_key()  # Load the key before starting the server
    run_server()
