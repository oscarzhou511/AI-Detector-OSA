import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel, AutoModelForCausalLM
from tqdm import tqdm
import json
import http.server
import socketserver
from http import HTTPStatus
import numpy as np

# --- Self-healing NLTK Data Check ---
import nltk
def check_and_download_nltk_data():
    required_packages = ['punkt', 'averaged_perceptron_tagger']
    print("Checking for required NLTK data...")
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
            print(f"  - NLTK package '{package}' already installed.")
        except LookupError:
            print(f"  - NLTK package '{package}' not found. Downloading...")
            nltk.download(package, quiet=True)
            print(f"  - '{package}' downloaded successfully.")
    print("NLTK data check complete.")
check_and_download_nltk_data()

import textstat
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.data import load as nltk_load # For span_tokenize

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
#  DETECTOR 1: Logit-based Highlighting (Unchanged)
# ==============================================================================
class LogitDetector:
    def __init__(self, model_name, device=DEVICE):
        print(f"Initializing LogitDetector with '{model_name}'...")
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sent_tokenizer = nltk_load('tokenizers/punkt/english.pickle')
        self.model.eval()
        print("LogitDetector initialized.")

    def _get_top_k_logits(self, input_ids_batch, top_k):
        with torch.no_grad():
            logits = self.model(input_ids_batch).logits
        last_token_logits = logits[:, -1, :]
        return torch.topk(last_token_logits, top_k, dim=-1).indices

    def _create_sentence_chunks(self, text, all_flagged_indices):
        flagged_indices_set = set(all_flagged_indices)
        sentence_spans = self.sent_tokenizer.span_tokenize(text)
        chunks = []
        global_token_offset = 0
        for start, end in sentence_spans:
            sentence_text = text[start:end]
            if not sentence_text.strip(): continue
            sentence_token_ids = self.tokenizer.encode(sentence_text, add_special_tokens=False)
            num_tokens_in_sentence = len(sentence_token_ids)
            if num_tokens_in_sentence == 0: continue
            flagged_count = sum(1 for i in range(num_tokens_in_sentence) if (global_token_offset + i) in flagged_indices_set)
            ai_percentage = (flagged_count / num_tokens_in_sentence) * 100
            sentence_type = 'AI' if ai_percentage > 60 else 'Human'
            chunks.append({"text": sentence_text, "type": sentence_type})
            global_token_offset += num_tokens_in_sentence
        return chunks

    def detect(self, text, top_k=10, batch_size=32, context_window=10):
        all_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(all_tokens) <= context_window:
            return [{"text": text, "type": "Human"}]
        input_sequences, target_tokens = [], []
        for i in range(len(all_tokens) - context_window):
            input_sequences.append(all_tokens[i : i + context_window])
            target_tokens.append(all_tokens[i + context_window])
        all_flagged_indices = []
        for i in tqdm(range(0, len(input_sequences), batch_size), desc="Logit Analysis", leave=False):
            batch_start, batch_end = i, i + batch_size
            input_ids_batch = torch.tensor(input_sequences[batch_start:batch_end]).to(self.device)
            top_k_preds_batch = self._get_top_k_logits(input_ids_batch, top_k)
            for j, top_k_preds in enumerate(top_k_preds_batch):
                if target_tokens[batch_start + j] in top_k_preds:
                    all_flagged_indices.append(batch_start + j + context_window)
        return self._create_sentence_chunks(text, all_flagged_indices)

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
    def detect(self, text, max_len=768):
        encoded = self.tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probability = torch.sigmoid(outputs["logits"]).item()
        return probability

# ==============================================================================
#  COMBINED DETECTOR: NOW WITH ENSEMBLE METHOD
# ==============================================================================
class CombinedDetector:
    def __init__(self):
        logit_model_path = "./models/distilgpt2"
        classifier_model_path = "./models/desklib-detector"
        print(f"Loading LogitDetector from local path: {logit_model_path}")
        self.logit_detector = LogitDetector(model_name=logit_model_path)
        print(f"Loading ClassifierDetector from local path: {classifier_model_path}")
        self.classifier_detector = ClassifierDetector(model_dir=classifier_model_path)
        self.linguistic_analyzer = LinguisticAnalyzer(perplexity_model=self.logit_detector.model, perplexity_tokenizer=self.logit_detector.tokenizer, device=DEVICE)

    def _calculate_ensemble_score(self, desklib_prob, highlight_chunks, linguistic_stats):
        """
        Calculates a final AI score using a weighted ensemble of three different methods.
        
        1. Desklib Classifier Score: Direct AI probability from the fine-tuned model. Highest weight.
        2. DistilGPT-2 Logit Score: Probability derived from the proportion of text flagged as AI.
        3. Linguistic Score: Score based on perplexity and "burstiness" (sentence length variation).
        
        Returns the final weighted probability and a dictionary of the component scores.
        """
        # --- Component 1: Desklib Score (Highest Weight) ---
        score_desklib = desklib_prob

        # --- Component 2: DistilGPT-2 Logit Score (Second Highest Weight) ---
        ai_char_count = sum(len(c['text']) for c in highlight_chunks if c['type'] == 'AI')
        total_char_count = sum(len(c['text']) for c in highlight_chunks)
        score_distilgpt2 = (ai_char_count / total_char_count) if total_char_count > 0 else 0.0

        # --- Component 3: Linguistic Feature Score (Third Highest Weight) ---
        score_linguistic = 0.0
        if linguistic_stats:
            # Perplexity component: Lower perplexity is more AI-like.
            # Map a perplexity range of [40, 100] to a score of [1.0, 0.0].
            perplexity = linguistic_stats.get('perplexity', 100) # Default to human-like
            perplexity_score = 1.0 - ((perplexity - 40) / (100 - 40))
            perplexity_score = np.clip(perplexity_score, 0, 1)

            # Burstiness component: Lower sentence length variation is more AI-like.
            # Map a variation range of [2, 8] to a score of [1.0, 0.0].
            sent_variation = linguistic_stats.get('sentence_length_variation', 8) # Default to human-like
            burstiness_score = 1.0 - ((sent_variation - 2) / (8 - 2))
            burstiness_score = np.clip(burstiness_score, 0, 1)

            score_linguistic = (perplexity_score + burstiness_score) / 2.0
        
        # --- Ensemble Weights ---
        W_DESKLIB = 0.60
        W_DISTILGPT2 = 0.25
        W_LINGUISTIC = 0.15

        # --- Final Weighted Score ---
        final_score = (W_DESKLIB * score_desklib) + \
                      (W_DISTILGPT2 * score_distilgpt2) + \
                      (W_LINGUISTIC * score_linguistic)
        
        component_scores = {
            "desklib_score": round(score_desklib, 4),
            "distilgpt2_logit_score": round(score_distilgpt2, 4),
            "linguistic_feature_score": round(score_linguistic, 4),
            "weights": {"desklib": W_DESKLIB, "distilgpt2": W_DISTILGPT2, "linguistic": W_LINGUISTIC}
        }
        return final_score, component_scores

    def detect(self, text):
        print("\n--- Starting New Analysis ---")
        
        # --- Step 1: Run all individual detectors ---
        desklib_prob = self.classifier_detector.detect(text)
        highlight_chunks = self.logit_detector.detect(text)
        linguistic_stats = self.linguistic_analyzer.analyze(text)
        
        # --- Step 2: Calculate the new ensemble score ---
        final_prob, component_scores = self._calculate_ensemble_score(
            desklib_prob=desklib_prob,
            highlight_chunks=highlight_chunks,
            linguistic_stats=linguistic_stats
        )

        # --- Step 3: Assemble the final result ---
        result = { 
            "overall_percentage": round(final_prob * 100, 2),
            "component_scores": component_scores,
            "chunks": highlight_chunks, 
            "linguistics": linguistic_stats 
        }
        return result

# ==============================================================================
#  SERVER SETUP (Unchanged)
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
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                body = json.loads(post_data)
                text_to_analyze = body.get('text')
                if not text_to_analyze or not isinstance(text_to_analyze, str) or len(text_to_analyze.strip()) == 0:
                    raise ValueError("Request body must contain non-empty 'text' field")
                
                results = DETECTOR_INSTANCE.detect(text_to_analyze)
                
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(results).encode('utf-8'))
            except Exception as e:
                print(f"Error processing request: {e}")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

def run_server(port=8000):
    with socketserver.TCPServer(("", port), AIRequestHandler) as httpd:
        print(f"Serving at http://localhost:{port}")
        print("Open the ai_detector_ui.html file in your browser to use the tool.")
        print("Press Ctrl+C to stop the server.")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()