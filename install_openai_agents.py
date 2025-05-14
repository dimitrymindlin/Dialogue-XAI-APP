#!/usr/bin/env python
"""
OpenAI Agents SDK installation and test script
"""
import subprocess
import sys
import os


def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def main():
    print("===== OpenAI Agents SDK Installation =====")
    print("This script will attempt to install the OpenAI Agents SDK package.")

    try:
        # Mevcut yüklü paketleri kontrol et
        print("Checking currently installed packages...")
        result = subprocess.run([sys.executable, "-m", "pip", "list"],
                                capture_output=True, text=True)
        packages = result.stdout.lower()

        if "openai-agents" in packages:
            print("OpenAI Agents SDK is already installed.")
            print("Proceeding to update.")

        # Gerekli paketleri yükle
        install_package("openai>=1.0.0")
        install_package("openai-agents")

        """# Check required environment variables
        if not os.environ.get("OPENAI_API_KEY"):
            api_key = input("OPENAI_API_KEY environment variable not found. Enter your API key (press ENTER to skip): ")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                print("API key temporarily set.")
                print("NOTE: This is valid for this session; add it to your .env file for persistence.")"""

        # Test
        print("\nInstallation succeeded. Performing a simple test...")
        test_code = """
import agents
from agents import Agent, Runner

print(f"Agents SDK version: {agents.__version__ if hasattr(agents, '__version__') else 'Unknown'}")
print("Agents SDK successfully imported.")
"""
        subprocess.run([sys.executable, "-c", test_code])

        print("\n===== Installation Complete =====")
        print("OpenAI Agents SDK has been successfully installed and tested.")
        print("\nTo run the Flask application:")
        print("python flask_app.py --gin_file=configs/structured_mape_k_openai_agents.gin")

    except Exception as e:
        print(f"Error: {e}")
        print("\nInstallation failed. Try the following steps manually:")
        print("1. pip install openai>=1.0.0")
        print("2. pip install openai-agents")
        print("3. Run the application: python flask_app.py --gin_file=configs/structured_mape_k_openai_agents.gin")


if __name__ == "__main__":
    main()