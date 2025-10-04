from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
import json
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import re
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# --- Simple in-memory database (no MongoDB required) ---
threat_db = []
print("✅ Using in-memory database (no MongoDB needed)")

# --- Configuration ---
VIRUSTOTAL_API_KEY = os.getenv('VIRUSTOTAL_API_KEY')

# --- ML Model Setup ---
class URLThreatDetector:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'url_length', 'host_length', 'path_length', 'num_digits', 'num_special_chars',
            'special_char_ratio', 'num_subdomains', 'has_https', 'has_ip', 'num_dots',
            'num_hyphens', 'num_underscores', 'suspicious_keyword_count', 'suspicious_tld'
        ]
        self.load_or_train_model()
    
    def extract_features(self, url):
        """Extract features from URL for ML model"""
        features = {}
        
        # URL Length Features
        features['url_length'] = len(url)
        features['host_length'] = len(urlparse(url).netloc)
        features['path_length'] = len(urlparse(url).path)
        
        # Character-Based Features
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_special_chars'] = sum(not c.isalnum() for c in url)
        features['special_char_ratio'] = features['num_special_chars'] / len(url) if len(url) > 0 else 0
        
        # Structural Features
        parsed = urlparse(url)
        features['num_subdomains'] = parsed.netloc.count('.')
        features['has_https'] = 1 if parsed.scheme == 'https' else 0
        features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc) else 0
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        
        # Keyword-Based Features
        suspicious_keywords = ['login', 'admin', 'php', 'exe', 'install', 'update', 'secure', 
                              'account', 'verify', 'banking', 'password', 'confirm', 'signin']
        features['suspicious_keyword_count'] = sum(1 for keyword in suspicious_keywords if keyword in url.lower())
        
        # TLD Features
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.xyz', '.top', '.club']
        features['suspicious_tld'] = 1 if any(tld in url.lower() for tld in suspicious_tlds) else 0
        
        return features

    def create_training_data(self):
        """Create synthetic training data for demo"""
        np.random.seed(42)
        n_samples = 300  # Smaller dataset for faster training
        
        X = []
        y = []
        
        # Benign URLs (class 0)
        for i in range(n_samples // 2):
            features = {
                'url_length': np.random.randint(20, 50),
                'host_length': np.random.randint(10, 30),
                'path_length': np.random.randint(5, 20),
                'num_digits': np.random.randint(0, 3),
                'num_special_chars': np.random.randint(2, 8),
                'special_char_ratio': np.random.uniform(0.05, 0.15),
                'num_subdomains': np.random.randint(1, 3),
                'has_https': 1,
                'has_ip': 0,
                'num_dots': np.random.randint(2, 4),
                'num_hyphens': np.random.randint(0, 2),
                'num_underscores': 0,
                'suspicious_keyword_count': np.random.randint(0, 1),
                'suspicious_tld': 0
            }
            X.append([features[feature] for feature in self.feature_names])
            y.append(0)  # Benign
        
        # Malicious URLs (class 1)
        for i in range(n_samples // 2):
            features = {
                'url_length': np.random.randint(50, 100),
                'host_length': np.random.randint(30, 60),
                'path_length': np.random.randint(20, 40),
                'num_digits': np.random.randint(3, 10),
                'num_special_chars': np.random.randint(8, 20),
                'special_char_ratio': np.random.uniform(0.15, 0.3),
                'num_subdomains': np.random.randint(3, 6),
                'has_https': np.random.choice([0, 1], p=[0.7, 0.3]),
                'has_ip': np.random.choice([0, 1], p=[0.3, 0.7]),
                'num_dots': np.random.randint(4, 8),
                'num_hyphens': np.random.randint(2, 6),
                'num_underscores': np.random.randint(1, 4),
                'suspicious_keyword_count': np.random.randint(2, 5),
                'suspicious_tld': np.random.choice([0, 1], p=[0.4, 0.6])
            }
            X.append([features[feature] for feature in self.feature_names])
            y.append(1)  # Malicious
        
        return np.array(X), np.array(y)

    def train_model(self):
        """Train the Random Forest model"""
        X, y = self.create_training_data()
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=50,  # Smaller for faster training
            max_depth=10,
            random_state=42
        )
        self.model.fit(X, y)
        
        # Save model
        os.makedirs('ml_model', exist_ok=True)
        joblib.dump(self.model, 'ml_model/url_detector_model.pkl')
        print("✅ Random Forest model trained and saved!")
        
        # Print feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        print("📊 Top 5 Feature Importance:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {feature}: {importance:.3f}")

    def load_model(self):
        """Load the trained model"""
        try:
            self.model = joblib.load('ml_model/url_detector_model.pkl')
            print("✅ ML model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Could not load ML model: {e}")
            self.model = None
            return False

    def load_or_train_model(self):
        """Load existing model or train new one"""
        if not self.load_model():
            print("🔄 Training new ML model...")
            self.train_model()

    def predict(self, url):
        """Predict if URL is malicious using ML model"""
        if self.model is None:
            return self._rule_based_fallback(url)
        
        try:
            # Extract features
            features_dict = self.extract_features(url)
            features = [features_dict[feature] for feature in self.feature_names]
            features_array = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(features_array)[0]
            probability = self.model.predict_proba(features_array)[0]
            
            confidence = probability[1] if prediction == 1 else probability[0]
            
            if prediction == 1:
                verdict = "Malicious"
                risk_level = "High"
            else:
                verdict = "Benign" 
                risk_level = "Low"
            
            return {
                'ml_verdict': verdict,
                'risk_level': risk_level,
                'confidence': round(float(confidence), 3),
                'features_analyzed': len(features),
                'model_used': 'Random Forest'
            }
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return self._rule_based_fallback(url)

    def _rule_based_fallback(self, url):
        """Fallback to rule-based prediction"""
        features = self.extract_features(url)
        
        # Calculate threat score
        score = 0
        score += min(features['url_length'] / 100, 1.0) * 0.2
        score += min(features['num_special_chars'] / 20, 1.0) * 0.3
        score += min(features['suspicious_keyword_count'] / 5, 1.0) * 0.3
        score += features['has_ip'] * 0.1
        score += features['suspicious_tld'] * 0.1
        
        confidence = min(score, 1.0)
        
        if confidence > 0.7:
            verdict = "Malicious"
            risk_level = "High"
        elif confidence > 0.4:
            verdict = "Suspicious"
            risk_level = "Medium"
        else:
            verdict = "Benign"
            risk_level = "Low"
        
        return {
            'ml_verdict': verdict,
            'risk_level': risk_level,
            'confidence': round(confidence, 3),
            'features_analyzed': len(features),
            'model_used': 'Rule-Based (Fallback)'
        }

# Initialize ML detector
ml_detector = URLThreatDetector()

# --- Helper Functions ---
def check_virustotal(ip_or_url):
    """Checks the IP/URL against the VirusTotal API."""
    if not VIRUSTOTAL_API_KEY or VIRUSTOTAL_API_KEY == "your_virustotal_key_here":
        # Return mock data for testing
        return {
            'found': True,
            'malicious_score': 0,
            'total_scanners': 75,
            'permalink': '#',
            'message': 'Using mock data - add real VirusTotal API key'
        }

    url = "https://www.virustotal.com/vtapi/v2/url/report"
    params = {'apikey': VIRUSTOTAL_API_KEY, 'resource': ip_or_url}
    
    try:
        response = requests.get(url, params=params)
        result = response.json()
        
        # Simplify the result for our dashboard
        if result.get('response_code') == 1:
            positives = result.get('positives', 0)
            total = result.get('total', 1)
            return {
                'found': True,
                'malicious_score': positives,
                'total_scanners': total,
                'permalink': result.get('permalink', '#')
            }
        else:
            return {'found': False, 'message': 'No data found in VirusTotal'}
    except Exception as e:
        return {'error': f'Failed to query VirusTotal: {str(e)}'}

def check_ml_model(ip_or_url):
    """Use the advanced ML detector"""
    try:
        result = ml_detector.predict(ip_or_url)
        print(f"🤖 ML Analysis: {result['ml_verdict']} (Confidence: {result['confidence']})")
        return result
    except Exception as e:
        print(f"❌ ML detection error: {e}")
        # Fallback to simple rule-based
        return {
            'ml_verdict': 'Error in ML model',
            'risk_level': 'Unknown',
            'confidence': 0.0,
            'model_used': 'Error - Using Fallback',
            'features_analyzed': 0
        }

# --- API Routes ---
@app.route('/api/check', methods=['POST'])
def check_threat():
    """Main API endpoint to check an IP/URL."""
    print("🔍 Received threat check request")
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        target = data.get('target', '').strip()
        print(f"🎯 Checking target: {target}")

        if not target:
            return jsonify({'error': 'No target provided'}), 400

        # 1. Check VirusTotal
        print("📡 Checking VirusTotal...")
        vt_result = check_virustotal(target)
        print(f"✅ VirusTotal result: {vt_result}")

        # 2. Check our ML Model
        print("🤖 Checking ML model...")
        ml_result = check_ml_model(target)
        print(f"✅ ML result: {ml_result}")

        # 3. Log this query to our simple database
        log_entry = {
            'target': target,
            'virustotal_result': vt_result,
            'ml_result': ml_result,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }
        threat_db.append(log_entry)
        print("💾 Saved to database")

        # 4. Combine and send the final result
        final_result = {
            'target': target,
            'virustotal': vt_result,
            'ml_model': ml_result,
            'overall_verdict': 'Malicious' if (vt_result.get('malicious_score', 0) > 2 or ml_result.get('confidence', 0) > 0.7) else 'Suspicious' if (vt_result.get('malicious_score', 0) > 0 or ml_result.get('confidence', 0) > 0.4) else 'Benign'
        }

        print(f"📊 Final verdict: {final_result['overall_verdict']}")
        return jsonify(final_result)
        
    except Exception as e:
        print(f"❌ Error in /api/check: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """API endpoint to get the history of checks."""
    try:
        # Return last 50 scans, newest first
        history = list(reversed(threat_db[-50:])) if threat_db else []
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get threat statistics"""
    try:
        total_scans = len(threat_db)
        malicious_scans = sum(1 for scan in threat_db if 
                            scan.get('virustotal_result', {}).get('malicious_score', 0) > 0 or 
                            scan.get('ml_result', {}).get('confidence', 0) > 0.6)
        
        return jsonify({
            'total': total_scans,
            'malicious': malicious_scans,
            'safe': total_scans - malicious_scans
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint to check if backend is working."""
    return jsonify({
        'status': 'healthy',
        'database': 'in-memory',
        'ml_model': 'loaded' if ml_detector.model is not None else 'training',
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'message': 'Backend is running with ML model'
    })

@app.route('/')
def home():
    return jsonify({
        'message': 'URL Threat Detector API is running!',
        'status': 'active',
        'ml_model': 'Random Forest Classifier',
        'endpoints': {
            'health_check': '/api/health',
            'check_threat': '/api/check (POST)',
            'scan_history': '/api/history (GET)',
            'statistics': '/api/statistics (GET)'
        }
    })

if __name__ == '__main__':
    print("🚀 Starting URL Threat Detector Backend...")
    print("📍 API running on: http://localhost:5000")
    print("🔧 Using in-memory database (perfect for demo)")
    print("🤖 ML Model: Random Forest Classifier")
    print("💡 Add VirusTotal API key in .env for real threat data")
    app.run(debug=True, port=5000)