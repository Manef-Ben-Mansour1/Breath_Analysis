from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import shutil
from werkzeug.utils import secure_filename
import librosa
import soundfile as sf
import numpy as np
import pywt
from collections import defaultdict
import logging
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Pour permettre les requêtes depuis React Native

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}

# Créer les dossiers nécessaires
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

class AudioProcessor:
    def __init__(self):
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            Shift(min_shift=-0.5, max_shift=0.5, p=0.5)
        ])

    def wavelet_denoise(self, audio_data, sr, wavelet='db4', level=2):
        """
        Appliquer le débruitage par ondelettes
        """
        try:
            # Décomposition en ondelettes
            coeffs = pywt.wavedec(audio_data, wavelet, level=level)
            
            # Estimation du seuil universel
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(audio_data)))
            
            # Application du seuillage aux coefficients de détail
            coeffs_thresh = [coeffs[0]]  # Garder les coefficients d'approximation
            for detail in coeffs[1:]:
                coeffs_thresh.append(pywt.threshold(detail, value=uthresh, mode='soft'))
            
            # Reconstruction du signal
            denoised_data = pywt.waverec(coeffs_thresh, wavelet)
            
            return denoised_data
        except Exception as e:
            logger.error(f"Erreur lors du débruitage: {e}")
            return audio_data

    def normalize_audio(self, audio_data):
        """
        Normaliser l'audio
        """
        try:
            if audio_data.size > 0:
                normalized = audio_data / max(1e-8, abs(audio_data).max())
                return normalized
            return audio_data
        except Exception as e:
            logger.error(f"Erreur lors de la normalisation: {e}")
            return audio_data

    def augment_audio(self, audio_data, sr, num_augmented=3):
        """
        Augmenter les données audio
        """
        augmented_samples = []
        try:
            for i in range(num_augmented):
                augmented = self.augment(samples=audio_data, sample_rate=sr)
                augmented_samples.append(augmented)
            return augmented_samples
        except Exception as e:
            logger.error(f"Erreur lors de l'augmentation: {e}")
            return []

    def get_audio_info(self, file_path):
        """
        Obtenir les informations de base du fichier audio
        """
        try:
            audio, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'samples': len(audio),
                'duration_minutes': duration / 60
            }
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse: {e}")
            return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialiser le processeur audio
processor = AudioProcessor()

@app.route('/health', methods=['GET'])
def health_check():
    """
    Vérifier l'état de l'API
    """
    return jsonify({'status': 'healthy', 'message': 'API is running'})

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Endpoint pour uploader un fichier audio
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Obtenir les informations du fichier
            audio_info = processor.get_audio_info(filepath)
            
            return jsonify({
                'message': 'Fichier uploadé avec succès',
                'filename': filename,
                'info': audio_info
            })
        else:
            return jsonify({'error': 'Type de fichier non autorisé'}), 400
            
    except Exception as e:
        logger.error(f"Erreur lors de l'upload: {e}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.route('/process', methods=['POST'])
def process_audio():
    """
    Endpoint pour traiter un fichier audio (débruitage + normalisation)
    """
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Nom de fichier requis'}), 400
        
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(input_path):
            return jsonify({'error': 'Fichier non trouvé'}), 404
        
        # Charger l'audio
        audio, sr = librosa.load(input_path, sr=None)
        
        # Appliquer le débruitage par ondelettes
        denoised_audio = processor.wavelet_denoise(audio, sr)
        
        # Normaliser l'audio
        normalized_audio = processor.normalize_audio(denoised_audio)
        
        # Sauvegarder le fichier traité
        processed_filename = f"processed_{filename}"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        sf.write(processed_path, normalized_audio, sr)
        
        return jsonify({
            'message': 'Fichier traité avec succès',
            'processed_filename': processed_filename,
            'original_duration': len(audio) / sr,
            'processed_duration': len(normalized_audio) / sr
        })
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {e}")
        return jsonify({'error': 'Erreur lors du traitement'}), 500

@app.route('/augment', methods=['POST'])
def augment_audio():
    """
    Endpoint pour augmenter les données audio
    """
    try:
        data = request.json
        filename = data.get('filename')
        num_augmented = data.get('num_augmented', 3)
        
        if not filename:
            return jsonify({'error': 'Nom de fichier requis'}), 400
        
        input_path = os.path.join(PROCESSED_FOLDER, filename)
        if not os.path.exists(input_path):
            return jsonify({'error': 'Fichier traité non trouvé'}), 404
        
        # Charger l'audio traité
        audio, sr = librosa.load(input_path, sr=None)
        
        # Générer les versions augmentées
        augmented_samples = processor.augment_audio(audio, sr, num_augmented)
        
        augmented_files = []
        base_name = filename.rsplit('.', 1)[0]
        
        for i, augmented in enumerate(augmented_samples):
            aug_filename = f"{base_name}_aug_{i}.wav"
            aug_path = os.path.join(PROCESSED_FOLDER, aug_filename)
            sf.write(aug_path, augmented, sr)
            augmented_files.append(aug_filename)
        
        return jsonify({
            'message': f'{len(augmented_files)} fichiers augmentés créés',
            'augmented_files': augmented_files
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de l\'augmentation: {e}")
        return jsonify({'error': 'Erreur lors de l\'augmentation'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """
    Endpoint pour analyser un fichier audio et extraire des caractéristiques
    """
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Nom de fichier requis'}), 400
        
        # Chercher le fichier dans les dossiers upload et processed
        file_path = None
        for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
            potential_path = os.path.join(folder, filename)
            if os.path.exists(potential_path):
                file_path = potential_path
                break
        
        if not file_path:
            return jsonify({'error': 'Fichier non trouvé'}), 404
        
        # Charger et analyser l'audio
        audio, sr = librosa.load(file_path, sr=None)
        
        # Extraire des caractéristiques de base
        duration = librosa.get_duration(y=audio, sr=sr)
        rms = librosa.feature.rms(y=audio)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        zero_crossings = librosa.feature.zero_crossing_rate(audio)[0]
        
        analysis = {
            'duration': duration,
            'duration_minutes': duration / 60,
            'sample_rate': sr,
            'samples': len(audio),
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'zero_crossing_rate_mean': float(np.mean(zero_crossings)),
            'zero_crossing_rate_std': float(np.std(zero_crossings))
        }
        
        return jsonify({
            'message': 'Analyse terminée',
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {e}")
        return jsonify({'error': 'Erreur lors de l\'analyse'}), 500

@app.route('/classify', methods=['POST'])
def classify_audio():
    """
    Endpoint pour la classification audio (à implémenter avec votre modèle)
    """
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Nom de fichier requis'}), 400
        
        # TODO: Implémenter la logique de classification avec votre modèle
        # Pour l'instant, retourner une classification fictive
        
        return jsonify({
            'message': 'Classification terminée',
            'prediction': {
                'class': 'classe_exemple',
                'confidence': 0.85,
                'probabilities': {
                    'classe_1': 0.85,
                    'classe_2': 0.10,
                    'classe_3': 0.05
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la classification: {e}")
        return jsonify({'error': 'Erreur lors de la classification'}), 500

@app.route('/files', methods=['GET'])
def list_files():
    """
    Lister tous les fichiers disponibles
    """
    try:
        uploaded_files = os.listdir(UPLOAD_FOLDER) if os.path.exists(UPLOAD_FOLDER) else []
        processed_files = os.listdir(PROCESSED_FOLDER) if os.path.exists(PROCESSED_FOLDER) else []
        
        return jsonify({
            'uploaded_files': uploaded_files,
            'processed_files': processed_files
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la liste des fichiers: {e}")
        return jsonify({'error': 'Erreur lors de la récupération des fichiers'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Fichier trop volumineux'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint non trouvé'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Erreur interne du serveur'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)