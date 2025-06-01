import requests
import json
import os

# Configuration
BASE_URL = 'http://localhost:5000'

class APITester:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
    
    def test_health(self):
        """Tester l'endpoint health"""
        try:
            response = requests.get(f'{self.base_url}/health')
            print(f"Health Check: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"Erreur health check: {e}")
            return False
    
    def test_upload(self, file_path):
        """Tester l'upload de fichier"""
        try:
            if not os.path.exists(file_path):
                print(f"Fichier non trouvé: {file_path}")
                return False
            
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f'{self.base_url}/upload', files=files)
            
            print(f"Upload: {response.status_code}")
            print(f"Response: {response.json()}")
            
            if response.status_code == 200:
                return response.json().get('filename')
            return False
            
        except Exception as e:
            print(f"Erreur upload: {e}")
            return False
    
    def test_process(self, filename):
        """Tester le traitement audio"""
        try:
            data = {'filename': filename}
            response = requests.post(
                f'{self.base_url}/process',
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Process: {response.status_code}")
            print(f"Response: {response.json()}")
            
            if response.status_code == 200:
                return response.json().get('processed_filename')
            return False
            
        except Exception as e:
            print(f"Erreur process: {e}")
            return False
    
    def test_analyze(self, filename):
        """Tester l'analyse audio"""
        try:
            data = {'filename': filename}
            response = requests.post(
                f'{self.base_url}/analyze',
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Analyze: {response.status_code}")
            print(f"Response: {response.json()}")
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"Erreur analyze: {e}")
            return False
    
    def test_augment(self, filename, num_augmented=2):
        """Tester l'augmentation audio"""
        try:
            data = {
                'filename': filename,
                'num_augmented': num_augmented
            }
            response = requests.post(
                f'{self.base_url}/augment',
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Augment: {response.status_code}")
            print(f"Response: {response.json()}")
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"Erreur augment: {e}")
            return False
    
    def test_classify(self, filename):
        """Tester la classification"""
        try:
            data = {'filename': filename}
            response = requests.post(
                f'{self.base_url}/classify',
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"Classify: {response.status_code}")
            print(f"Response: {response.json()}")
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"Erreur classify: {e}")
            return False
    
    def test_list_files(self):
        """Tester la liste des fichiers"""
        try:
            response = requests.get(f'{self.base_url}/files')
            print(f"List Files: {response.status_code}")
            print(f"Response: {response.json()}")
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"Erreur list files: {e}")
            return False
    
    def run_full_test(self, test_file_path):
        """Exécuter tous les tests dans l'ordre"""
        print("=== Test complet de l'API ===")
        
        # 1. Health check
        print("\n1. Test Health Check")
        if not self.test_health():
            print("❌ Health check échoué")
            return
        print("✅ Health check réussi")
        
        # 2. Upload
        print("\n2. Test Upload")
        filename = self.test_upload(test_file_path)
        if not filename:
            print("❌ Upload échoué")
            return
        print(f"✅ Upload réussi: {filename}")
        
        # 3. Analyze original
        print("\n3. Test Analyze (original)")
        if not self.test_analyze(filename):
            print("❌ Analyse échouée")
        else:
            print("✅ Analyse réussie")
        
        # 4. Process
        print("\n4. Test Process")
        processed_filename = self.test_process(filename)
        if not processed_filename:
            print("❌ Traitement échoué")
            return
        print(f"✅ Traitement réussi: {processed_filename}")
        
        # 5. Analyze processed
        print("\n5. Test Analyze (processed)")
        if not self.test_analyze(processed_filename):
            print("❌ Analyse du fichier traité échouée")
        else:
            print("✅ Analyse du fichier traité réussie")
        
        # 6. Augment
        print("\n6. Test Augment")
        if not self.test_augment(processed_filename, 2):
            print("❌ Augmentation échouée")
        else:
            print("✅ Augmentation réussie")
        
        # 7. Classify
        print("\n7. Test Classify")
        if not self.test_classify(processed_filename):
            print("❌ Classification échouée")
        else:
            print("✅ Classification réussie")
        
        # 8. List files
        print("\n8. Test List Files")
        if not self.test_list_files():
            print("❌ Liste des fichiers échouée")
        else:
            print("✅ Liste des fichiers réussie")
        
        print("\n=== Test complet terminé ===")

def main():
    """Fonction principale pour exécuter les tests"""
    tester = APITester()
    
    # Vous devez fournir le chemin vers un fichier audio de test
    test_file = input("Entrez le chemin vers un fichier audio de test (.wav): ")
    
    if not test_file:
        print("Aucun fichier spécifié. Test simple sans fichier...")
        tester.test_health()
        tester.test_list_files()
    else:
        tester.run_full_test(test_file)

if __name__ == "__main__":
    main()