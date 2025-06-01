#!/usr/bin/env python3
"""
Script de démarrage pour le serveur Flask
"""

import os
import sys
from app import app
from config import config

def create_directories():
    """Créer les dossiers nécessaires"""
    directories = ['uploads', 'processed', 'models', 'logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Dossier créé: {directory}")
        else:
            print(f"📁 Dossier existe: {directory}")

def check_dependencies():
    """Vérifier que toutes les dépendances sont installées"""
    required_modules = [
        'flask', 'librosa', 'soundfile', 'numpy', 
        'pywt', 'audiomentations', 'flask_cors'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ Module trouvé: {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ Module manquant: {module}")
    
    if missing_modules:
        print(f"\n⚠️  Modules manquants: {', '.join(missing_modules)}")
        print("Installez-les avec: pip install -r requirements.txt")
        return False
    
    print("✅ Toutes les dépendances sont installées")
    return True

def main():
    """Fonction principale"""
    print("🚀 Démarrage du serveur Flask Audio API")
    print("=" * 50)
    
    # Vérifier les dépendances
    print("\n📦 Vérification des dépendances...")
    if not check_dependencies():
        sys.exit(1)
    
    # Créer les dossiers
    print("\n📁 Création des dossiers...")
    create_directories()
    
    # Configuration
    env = os.environ.get('FLASK_ENV', 'development')
    config_obj = config.get(env, config['default'])
    
    print(f"\n⚙️  Configuration: {env}")
    print(f"🌐 Host: {config_obj.HOST}")
    print(f"🔌 Port: {config_obj.PORT}")
    print(f"🐛 Debug: {config_obj.DEBUG}")
    
    # Démarrer le serveur
    print(f"\n🎵 Serveur Audio API démarré sur http://{config_obj.HOST}:{config_obj.PORT}")
    print("📋 Endpoints disponibles:")
    print("   GET  /health          - Vérification de l'état")
    print("   POST /upload          - Upload de fichier audio")
    print("   POST /process         - Traitement audio (débruitage + normalisation)")
    print("   POST /analyze         - Analyse des caractéristiques audio")
    print("   POST /augment         - Augmentation des données")
    print("   POST /classify        - Classification audio")
    print("   GET  /files           - Liste des fichiers")
    print("\n🛑 Appuyez sur Ctrl+C pour arrêter le serveur")
    print("=" * 50)
    
    try:
        app.run(
            host=config_obj.HOST,
            port=config_obj.PORT,
            debug=config_obj.DEBUG,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Serveur arrêté par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur lors du démarrage: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()