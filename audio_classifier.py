#!/usr/bin/env python3
"""
Flask API for Audio Classification - FIXED JSON SERIALIZATION VERSION
Receives audio from React Native app and returns disease prediction
"""

import os
import tempfile
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import pickle
import tensorflow as tf
from tensorflow import keras
import warnings
import subprocess
import shutil
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, origins=["*"], allow_headers=["Content-Type", "Authorization", "Accept"], methods=["GET", "POST", "OPTIONS"],expose_headers=["Content-Type"])  # Enable CORS for React Native

class AudioClassifier:
    def __init__(self, model_path="resnet_audio_classifier.h5", encoder_path="label_encoder.pkl"):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.label_encoder = None
        
        # Training parameters - MUST MATCH EXACTLY
        self.IMG_HEIGHT = 224
        self.IMG_WIDTH = 224
        self.n_mels = 128  # This was used in training!
        
        # Load model and encoder
        self._load_components()
    
    def _load_components(self):
        """Load the trained model and label encoder"""
        try:
            # Load model
            print("Loading model...")
            self.model = keras.models.load_model(self.model_path)
            print(f"✓ Model loaded: {self.model_path}")
            
            # Load label encoder
            print("Loading label encoder...")
            with open(self.encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"✓ Label encoder loaded with {len(self.label_encoder.classes_)} classes")
            print(f"Classes: {', '.join(self.label_encoder.classes_)}")
            
        except Exception as e:
            print(f"Error loading components: {e}")
            raise
    
    def _convert_audio_to_wav(self, input_path):
        """
        Convert audio file to WAV format using ffmpeg or fallback methods
        This fixes the m4a processing issue
        """
        try:
            # Create temporary WAV file
            temp_wav = input_path.replace(os.path.splitext(input_path)[1], '_converted.wav')
            
            print(f"🔄 Converting {input_path} to WAV format...")
            
            # Method 1: Try using ffmpeg (if available)
            if shutil.which('ffmpeg'):
                try:
                    subprocess.run([
                        'ffmpeg', '-i', input_path, 
                        '-ar', '44100',  # Sample rate
                        '-ac', '1',      # Mono
                        '-y',            # Overwrite
                        temp_wav
                    ], check=True, capture_output=True)
                    print(f"✅ Converted using ffmpeg: {temp_wav}")
                    return temp_wav
                except subprocess.CalledProcessError as e:
                    print(f"⚠️ FFmpeg failed: {e}")
            
            # Method 2: Try using librosa directly with original file
            try:
                print("🔄 Trying direct librosa conversion...")
                y, sr = librosa.load(input_path, sr=44100, mono=True)
                # Save as WAV
                import soundfile as sf
                sf.write(temp_wav, y, sr)
                print(f"✅ Converted using librosa: {temp_wav}")
                return temp_wav
            except Exception as e:
                print(f"⚠️ Librosa direct conversion failed: {e}")
            
            # Method 3: Try different librosa parameters
            try:
                print("🔄 Trying librosa with different parameters...")
                y, sr = librosa.load(input_path, sr=None)  # Keep original sample rate
                # Resample to 44100 if needed
                if sr != 44100:
                    y = librosa.resample(y, orig_sr=sr, target_sr=44100)
                    sr = 44100
                
                # Convert to mono if stereo
                if len(y.shape) > 1:
                    y = librosa.to_mono(y)
                
                # Save as WAV
                import soundfile as sf
                sf.write(temp_wav, y, sr)
                print(f"✅ Converted with resampling: {temp_wav}")
                return temp_wav
            except Exception as e:
                print(f"⚠️ Librosa resampling failed: {e}")
            
            # If all methods fail, return original file path
            print("⚠️ All conversion methods failed, trying with original file...")
            return input_path
            
        except Exception as e:
            print(f"❌ Audio conversion error: {e}")
            return input_path
    
    def wav_to_mel_spectrogram(self, file_path):
        """
        ENHANCED version with better error handling and audio format support
        """
        try:
            print(f"🎵 Processing audio file: {os.path.basename(file_path)}")
            print(f"📁 File size: {os.path.getsize(file_path)} bytes")
            
            # Convert to WAV if needed
            processed_file = file_path
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension in ['.m4a', '.aac', '.mp3', '.flac']:
                processed_file = self._convert_audio_to_wav(file_path)
            
            # Load audio with error handling
            print(f"🔄 Loading audio with librosa...")
            try:
                # Try with sr=None first (keep original sample rate)
                y, sr = librosa.load(processed_file, sr=None)
                print(f"✅ Audio loaded: {len(y)} samples at {sr}Hz")
            except Exception as e1:
                print(f"⚠️ Failed with sr=None, trying sr=22050: {e1}")
                try:
                    # Fallback to 22050 Hz
                    y, sr = librosa.load(processed_file, sr=22050)
                    print(f"✅ Audio loaded with fallback: {len(y)} samples at {sr}Hz")
                except Exception as e2:
                    print(f"⚠️ Failed with sr=22050, trying sr=16000: {e2}")
                    # Last resort: 16kHz
                    y, sr = librosa.load(processed_file, sr=16000)
                    print(f"✅ Audio loaded with last resort: {len(y)} samples at {sr}Hz")
            
            # Check if audio is valid
            if len(y) == 0:
                raise ValueError("Audio file is empty or corrupted")
            
            # Normalize audio
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            print(f"🎛️ Audio stats: min={np.min(y):.3f}, max={np.max(y):.3f}, mean={np.mean(y):.3f}")
            
            # Generate mel spectrogram with same parameters as training
            print(f"🔄 Generating mel spectrogram (n_mels={self.n_mels})...")
            mels = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=self.n_mels,
                n_fft=2048,      # Default FFT window size
                hop_length=512,  # Default hop length
                fmax=sr//2       # Maximum frequency
            )
            
            # Check mel spectrogram
            if mels.size == 0:
                raise ValueError("Generated mel spectrogram is empty")
            
            print(f"✅ Mel spectrogram shape: {mels.shape}")
            
            # Convert to dB - same as training
            mels_db = librosa.power_to_db(mels, ref=np.max)
            print(f"✅ Mel spectrogram in dB: min={np.min(mels_db):.1f}, max={np.max(mels_db):.1f}")
            
            # Clean up temporary file if created
            if processed_file != file_path and os.path.exists(processed_file):
                try:
                    os.remove(processed_file)
                    print(f"🧹 Cleaned up temporary file: {processed_file}")
                except:
                    pass
            
            return mels_db
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
            
            # Clean up temporary file if created
            if 'processed_file' in locals() and processed_file != file_path and os.path.exists(processed_file):
                try:
                    os.remove(processed_file)
                except:
                    pass
            
            return None
    
    def _process_mel_spectrogram_exact_training(self, mels_db):
        """
        EXACT same processing as in training with enhanced error handling
        """
        try:
            print(f"🔄 Processing mel spectrogram for model input...")
            print(f"📊 Input shape: {mels_db.shape}")
            
            # Step 1: Add new axis (same as mel[..., np.newaxis])
            mel_with_axis = mels_db[..., np.newaxis]
            print(f"📊 After adding axis: {mel_with_axis.shape}")
            
            # Step 2: Use tf.image.resize exactly as in training
            mel_resized = tf.image.resize(mel_with_axis, (self.IMG_HEIGHT, self.IMG_WIDTH)).numpy()
            print(f"📊 After resize: {mel_resized.shape}")
            
            # Step 3: Repeat 3 times for RGB (same as np.repeat(mel, 3, axis=-1))
            mel_rgb = np.repeat(mel_resized, 3, axis=-1)
            print(f"📊 Final RGB shape: {mel_rgb.shape}")
            
            # Verify final shape
            expected_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, 3)
            if mel_rgb.shape != expected_shape:
                raise ValueError(f"Final shape {mel_rgb.shape} doesn't match expected {expected_shape}")
            
            print(f"✅ Mel spectrogram processed successfully")
            return mel_rgb
            
        except Exception as e:
            print(f"❌ Error processing mel spectrogram: {e}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return None
    
    def classify_single_prediction(self, audio_path):
        """
        Classify an audio file and return only the most probable disease
        FIXED: Proper JSON serialization for all numpy types
        """
        
        # Check if file exists
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
        
        print(f"🩺 Classifying: {os.path.basename(audio_path)}")
        
        try:
            # Step 1: Convert to mel spectrogram
            print("📊 Step 1: Converting to mel spectrogram...")
            mels_db = self.wav_to_mel_spectrogram(audio_path)
            if mels_db is None:
                return {"error": "Failed to process audio file - could not generate mel spectrogram"}
            
            # Step 2: Process exactly as in training
            print("🔄 Step 2: Processing for model input...")
            spectrogram_rgb = self._process_mel_spectrogram_exact_training(mels_db)
            if spectrogram_rgb is None:
                return {"error": "Failed to process mel spectrogram for model input"}
            
            # Step 3: Prepare batch input
            print("📦 Step 3: Preparing batch input...")
            input_data = np.expand_dims(spectrogram_rgb, axis=0)
            print(f"📊 Model input shape: {input_data.shape}")
            
            # Step 4: Make prediction
            print("🤖 Step 4: Making prediction...")
            predictions = self.model.predict(input_data, verbose=0)
            print(f"📊 Prediction shape: {predictions.shape}")
            print(f"📊 Prediction range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
            
            # Step 5: Get the most probable prediction
            predicted_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            predicted_disease = self.label_encoder.classes_[predicted_idx]
            
            print(f"🎯 Predicted index: {predicted_idx}")
            print(f"🎯 Predicted disease: {predicted_disease}")
            print(f"🎯 Confidence: {confidence:.4f}")
            
            # Calculate prediction entropy (uncertainty measure)
            entropy = -np.sum(predictions[0] * np.log(predictions[0] + 1e-8))
            
            # Determine confidence level
            if confidence > 0.8:
                confidence_level = "Very High"
            elif confidence > 0.6:
                confidence_level = "High"
            elif confidence > 0.4:
                confidence_level = "Medium"
            elif confidence > 0.2:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"
            
            # FIXED: Convert all numpy types to native Python types for JSON serialization
            result = {
                "success": True,
                "predicted_disease": str(predicted_disease),  # Ensure string
                "confidence": float(confidence),  # Convert numpy.float32 to Python float
                "confidence_percent": f"{float(confidence)*100:.1f}%",
                "confidence_level": str(confidence_level),
                "is_reliable": bool(confidence >= 0.5),  # Convert numpy.bool_ to Python bool
                "prediction_entropy": float(entropy),  # Convert numpy.float64 to Python float
                "audio_file": str(os.path.basename(audio_path))  # Ensure string
            }
            
            print(f"✅ Classification successful: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Classification error: {e}")
            print(f"🔍 Full traceback:")
            traceback.print_exc()
            return {"error": f"Classification failed: {str(e)}"}

# Initialize classifier globally
print("Initializing Audio Classifier...")
try:
    classifier = AudioClassifier()
    print("✓ Classifier initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize classifier: {e}")
    classifier = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if classifier is None:
        return jsonify({
            "status": "error",
            'port': 5000,
            "message": "Classifier not initialized"
        }), 500
    
    return jsonify({
        "status": "healthy",
        "message": "Audio Classification API is running",
        "model_loaded": classifier.model is not None,
        "encoder_loaded": classifier.label_encoder is not None,
        "available_classes": len(classifier.label_encoder.classes_) if classifier.label_encoder else 0
    })

@app.route('/classify', methods=['POST'])
def classify_audio():
    """
    Main endpoint to classify audio files
    Accepts multipart/form-data with 'audio' file
    Returns single most probable disease prediction
    FIXED: Proper error handling and JSON serialization
    """
    
    # Check if classifier is initialized
    if classifier is None:
        return jsonify({
            "success": False,
            "error": "Audio classifier not initialized"
        }), 500
    
    # Check if file is present in request
    if 'audio' not in request.files:
        return jsonify({
            "success": False,
            "error": "No audio file provided. Please send audio file with key 'audio'"
        }), 400
    
    audio_file = request.files['audio']
    
    # Check if file is selected
    if audio_file.filename == '':
        return jsonify({
            "success": False,
            "error": "No audio file selected"
        }), 400
    
    # Check file extension
    allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    file_extension = os.path.splitext(audio_file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        return jsonify({
            "success": False,
            "error": f"Unsupported file format: {file_extension}. Supported formats: {', '.join(allowed_extensions)}"
        }), 400
    
    # Save uploaded file temporarily
    temp_dir = tempfile.gettempdir()
    temp_filename = f"audio_classify_{os.urandom(8).hex()}{file_extension}"
    temp_filepath = os.path.join(temp_dir, temp_filename)
    
    try:
        # Save the uploaded file
        print(f"💾 Saving uploaded file to: {temp_filepath}")
        audio_file.save(temp_filepath)
        
        # Verify file was saved and has content
        if not os.path.exists(temp_filepath) or os.path.getsize(temp_filepath) == 0:
            return jsonify({
                "success": False,
                "error": "Failed to save audio file or file is empty"
            }), 500
        
        file_size = os.path.getsize(temp_filepath)
        print(f"✅ File saved successfully. Size: {file_size} bytes")
        
        # Additional file validation
        if file_size < 1000:  # Less than 1KB is probably too small
            print(f"⚠️ Warning: File size is very small ({file_size} bytes)")
        
        # Classify the audio
        print("🚀 Starting classification...")
        result = classifier.classify_single_prediction(temp_filepath)
        
        # Check if classification was successful
        if "error" in result:
            print(f"❌ Classification failed: {result['error']}")
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500
        
        # FIXED: Ensure all values are JSON serializable before returning
        # This should already be handled in classify_single_prediction, but double-check
        json_safe_result = {}
        for key, value in result.items():
            if isinstance(value, (np.integer, np.floating, np.bool_)):
                # Convert numpy types to native Python types
                if isinstance(value, np.bool_):
                    json_safe_result[key] = bool(value)
                elif isinstance(value, np.integer):
                    json_safe_result[key] = int(value)
                elif isinstance(value, np.floating):
                    json_safe_result[key] = float(value)
            else:
                json_safe_result[key] = value
        
        # Return successful result
        print(f"🎉 Classification successful: {json_safe_result['predicted_disease']} ({json_safe_result['confidence_percent']})")
        print("Final result:", json_safe_result)
        return jsonify(json_safe_result), 200
        
    except Exception as e:
        print(f"❌ Error during classification: {e}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Server error during classification: {str(e)}"
        }), 500
        
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
                print(f"🧹 Cleaned up temporary file: {temp_filepath}")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up temporary file {temp_filepath}: {e}")

@app.route('/classes', methods=['GET'])
def get_available_classes():
    """Get list of available disease classes"""
    if classifier is None or classifier.label_encoder is None:
        return jsonify({
            "success": False,
            "error": "Classifier not initialized"
        }), 500
    
    return jsonify({
        "success": True,
        "classes": classifier.label_encoder.classes_.tolist(),
        "total_classes": len(classifier.label_encoder.classes_)
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "success": False,
        "error": "File too large. Maximum file size exceeded."
    }), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    # Configuration
    HOST = '0.0.0.0'  # Listen on all interfaces
    PORT = 5002
    DEBUG = True
    
    # Set maximum file size (16MB)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    
    print("="*60)
    print("🎵 AUDIO CLASSIFICATION API SERVER - JSON SERIALIZATION FIXED")
    print("="*60)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Debug: {DEBUG}")
    print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")
    
    # Check for optional dependencies
    print("\n🔍 Checking dependencies:")
    try:
        import soundfile
        print("✅ soundfile available")
    except ImportError:
        print("⚠️ soundfile not available (install with: pip install soundfile)")
    
    if shutil.which('ffmpeg'):
        print("✅ ffmpeg available")
    else:
        print("⚠️ ffmpeg not available (install from https://ffmpeg.org/)")
    
    if classifier:
        print(f"\n🤖 Model info:")
        print(f"Available classes: {len(classifier.label_encoder.classes_)}")
        print(f"Classes: {', '.join(classifier.label_encoder.classes_)}")
    
    print("\n📡 API Endpoints:")
    print(f"  Health Check: http://{HOST}:{PORT}/health")
    print(f"  Classify Audio: http://{HOST}:{PORT}/classify (POST)")
    print(f"  Get Classes: http://{HOST}:{PORT}/classes")
    
    print("\n🔧 Usage from React Native:")
    print("  Make sure to update API_CONFIG.baseUrl in your React Native app")
    print(f"  Example: baseUrl: 'http://YOUR_SERVER_IP:{PORT}'")
    
    print("\n🚀 Starting server...")
    print("="*60)
    
    app.run(host=HOST, port=PORT, debug=DEBUG, use_reloader=False)