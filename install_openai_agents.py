#!/usr/bin/env python
"""
OpenAI Agents SDK yükleme ve test scripti
"""
import subprocess
import sys
import os

def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("===== OpenAI Agents SDK Yükleme =====")
    print("Bu script OpenAI Agents SDK paketini yüklemeye çalışacak.")
    
    try:
        # Mevcut yüklü paketleri kontrol et
        print("Mevcut yüklü paketleri kontrol ediliyor...")
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                                capture_output=True, text=True)
        packages = result.stdout.lower()
        
        if "openai-agents" in packages:
            print("OpenAI Agents SDK zaten yüklü.")
            print("Güncellemek için devam edilecek.")
        
        # Gerekli paketleri yükle
        install_package("openai>=1.0.0")
        install_package("openai-agents")
        
        # Gerekli ortam değişkenlerini kontrol et
        if not os.environ.get("OPENAI_API_KEY"):
            api_key = input("OPENAI_API_KEY çevre değişkeni bulunamadı. API anahtarınızı girin (ENTER ile geçin): ")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                print("API anahtarı geçici olarak ayarlandı.")
                print("NOT: Bu oturum için geçerlidir, kalıcı olması için .env dosyasına ekleyin.")
        
        # Test et
        print("\nYükleme başarılı. Basit bir test yapılıyor...")
        test_code = """
import agents
from agents import Agent, Runner

print(f"Agents SDK version: {agents.__version__ if hasattr(agents, '__version__') else 'Unknown'}")
print("Agents SDK successfully imported.")
"""
        subprocess.run([sys.executable, "-c", test_code])
        
        print("\n===== Yükleme Tamamlandı =====")
        print("OpenAI Agents SDK başarıyla yüklendi ve test edildi.")
        print("\nFlask uygulamasını çalıştırmak için:")
        print("python flask_app.py --gin_file=configs/structured_mape_k_openai_agents.gin")
        
    except Exception as e:
        print(f"Hata: {e}")
        print("\nYükleme başarısız oldu. Aşağıdaki adımları manuel olarak deneyin:")
        print("1. pip install openai>=1.0.0")
        print("2. pip install openai-agents")
        print("3. Uygulamayı çalıştırın: python flask_app.py --gin_file=configs/structured_mape_k_openai_agents.gin")

if __name__ == "__main__":
    main() 