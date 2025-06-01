import numpy as np
import pywt
import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from collections import defaultdict
import os
import logging

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """
    Classe pour le preprocessing des fichiers audio
    """
    
    def __init__(self, wavelet='db4', level=2, sr=22050):
        self.wavelet = wavelet
        self.level = level
        self.sr = sr
        
        # Configuration de l'augmentation
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            Shift(min_shift=-0.5, max_shift=0.5, p=0.5)
        ])
    
    def load_audio(self, file_path, sr=None):
        """
        Charger un fichier audio
        """
        try:
            if sr is None:
                sr = self.sr
            audio, sample_rate = librosa.load(file_path, sr=sr)
            return audio, sample_rate
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {file_path}: {e}")
            return None, None
    
    def wavelet_denoise(self, audio_data, wavelet=None, level=None):
        """
        Appliquer le débruitage par ondelettes
        """
        if wavelet is None:
            wavelet = self.wavelet
        if level is None:
            level = self.level
            
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
            
            # S'assurer que la longueur reste la même
            if len(denoised_data) != len(audio_data):
                denoised_data = denoised_data[:len(audio_data)]
                
            return denoised_data
            
        except Exception as e:
            logger.error(f"Erreur lors du débruitage par ondelettes: {e}")
            return audio_data
    
    def normalize_audio(self, audio_data):
        """
        Normaliser l'audio (peak normalization)
        """
        try:
            if audio_data.size > 0:
                max_val = np.max(np.abs(audio_data))
                if max_val > 1e-8:  # Éviter la division par zéro
                    normalized = audio_data / max_val
                else:
                    normalized = audio_data
                return normalized
            return audio_data
        except Exception as e:
            logger.error(f"Erreur lors de la normalisation: {e}")
            return audio_data
    
    def rms_normalize(self, audio_data, target_rms=0.1):
        """
        Normalisation RMS
        """
        try:
            rms = np.sqrt(np.mean(audio_data**2))
            if rms > 1e-8:
                normalized = audio_data * (target_rms / rms)
                return normalized
            return audio_data
        except Exception as e:
            logger.error(f"Erreur lors de la normalisation RMS: {e}")
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
    
    def extract_features(self, audio_data, sr):
        """
        Extraire des caractéristiques audio
        """
        try:
            features = {}
            
            # Caractéristiques temporelles
            features['duration'] = len(audio_data) / sr
            features['rms'] = np.sqrt(np.mean(audio_data**2))
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
            
            # Caractéristiques spectrales
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0])
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0])
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0])
            
            # MFCC
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}'] = np.mean(mfccs[i])
            
            # Chroma
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            
            # Contraste spectral
            contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
            features['spectral_contrast'] = np.mean(contrast)
            
            return features
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des caractéristiques: {e}")
            return {}
    
    def process_audio_file(self, input_path, output_path=None, 
                          denoise=True, normalize=True, extract_features=False):
        """
        Traiter un fichier audio complet
        """
        try:
            # Charger l'audio
            audio, sr = self.load_audio(input_path)
            if audio is None:
                return None
            
            original_audio = audio.copy()
            
            # Débruitage par ondelettes
            if denoise:
                audio = self.wavelet_denoise(audio)
                logger.info("Débruitage appliqué")
            
            # Normalisation
            if normalize:
                audio = self.normalize_audio(audio)
                logger.info("Normalisation appliquée")
            
            # Sauvegarder le fichier traité
            if output_path:
                sf.write(output_path, audio, sr)
                logger.info(f"Fichier sauvegardé: {output_path}")
            
            result = {
                'original_audio': original_audio,
                'processed_audio': audio,
                'sample_rate': sr,
                'success': True
            }
            
            # Extraire les caractéristiques si demandé
            if extract_features:
                result['features'] = self.extract_features(audio, sr)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {input_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    def batch_process(self, input_dir, output_dir, file_extension='.wav'):
        """
        Traitement par lots de fichiers audio
        """
        processed_files = []
        failed_files = []
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(file_extension):
                    input_path = os.path.join(root, file)
                    output_path = os.path.join(output_dir, f"processed_{file}")
                    
                    result = self.process_audio_file(input_path, output_path)
                    
                    if result and result.get('success'):
                        processed_files.append(file)
                        logger.info(f"Traité avec succès: {file}")
                    else:
                        failed_files.append(file)
                        logger.error(f"Échec du traitement: {file}")
        
        return {
            'processed_files': processed_files,
            'failed_files': failed_files,
            'total_processed': len(processed_files),
            'total_failed': len(failed_files)
        }
    
    def analyze_dataset(self, data_path, annotations_df=None):
        """
        Analyser un dataset audio complet
        """
        durations = defaultdict(float)
        file_counts = defaultdict(int)
        
        if annotations_df is not None:
            # Analyse basée sur les annotations
            for _, row in annotations_df.iterrows():
                prefix = str(row['id'])
                matching_files = []
                
                for root, dirs, files in os.walk(data_path):
                    for file in files:
                        if file.endswith('.wav') and file.startswith(prefix):
                            matching_files.append(os.path.join(root, file))
                
                for file_path in matching_files:
                    if os.path.exists(file_path):
                        try:
                            audio, sr = self.load_audio(file_path)
                            if audio is not None:
                                duration = len(audio) / sr
                                durations[row['label']] += duration
                                file_counts[row['label']] += 1
                        except Exception as e:
                            logger.error(f"Erreur avec le fichier {file_path}: {e}")
        else:
            # Analyse simple de tous les fichiers
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.wav'):
                        file_path = os.path.join(root, file)
                        try:
                            audio, sr = self.load_audio(file_path)
                            if audio is not None:
                                duration = len(audio) / sr
                                durations['all_files'] += duration
                                file_counts['all_files'] += 1
                        except Exception as e:
                            logger.error(f"Erreur avec le fichier {file_path}: {e}")
        
        # Convertir en minutes et créer le résumé
        analysis = {
            'durations_seconds': dict(durations),
            'durations_minutes': {k: v/60 for k, v in durations.items()},
            'file_counts': dict(file_counts),
            'total_duration_minutes': sum(durations.values()) / 60,
            'total_files': sum(file_counts.values())
        }
        
        return analysis