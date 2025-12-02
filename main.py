"""
Point d'entrée principal de l'assistant IA.

Pour l'instant, c'est un simple chatbot en ligne de commande.
On ajoutera l'interface web plus tard.

Usage:
    python main.py
"""

import asyncio
import yaml
from pathlib import Path

from src.llm import OllamaLLM
from src.llm.base import Message


def load_config() -> dict:
    """Charge la configuration depuis config.yaml"""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def main():
    """
    Boucle principale du chatbot.
    
    C'est un REPL (Read-Eval-Print-Loop):
    1. Lire l'entrée utilisateur
    2. Envoyer au LLM
    3. Afficher la réponse
    4. Répéter
    """
    # Charger la configuration
    config = load_config()
    llm_config = config["llm"]["ollama"]
    character = config["character"]
    
    print("=" * 50)
    print(f"🤖 {character['name']} - Assistant IA")
    print("=" * 50)
    print("Tape 'quit' pour quitter, 'clear' pour effacer l'historique")
    print()
    
    # Créer le client LLM
    llm = OllamaLLM(
        model=llm_config["model"],
        base_url=llm_config["base_url"]
    )
    
    # Historique de la conversation
    # On commence avec le system prompt qui définit la personnalité
    messages: list[Message] = [
        Message(role="system", content=character["system_prompt"])
    ]
    
    try:
        while True:
            # 1. Lire l'entrée utilisateur
            try:
                user_input = input("\n👤 Toi: ").strip()
            except EOFError:
                break
                
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("\n👋 À bientôt !")
                break
            if user_input.lower() == "clear":
                messages = [Message(role="system", content=character["system_prompt"])]
                print("🗑️  Historique effacé !")
                continue
            
            # 2. Ajouter le message utilisateur à l'historique
            messages.append(Message(role="user", content=user_input))
            
            # 3. Obtenir la réponse du LLM (avec streaming)
            print(f"\n🤖 {character['name']}: ", end="", flush=True)
            
            full_response = ""
            async for chunk in llm.chat_stream(messages):
                print(chunk, end="", flush=True)  # Affiche au fur et à mesure
                full_response += chunk
            print()  # Nouvelle ligne à la fin
            
            # 4. Ajouter la réponse à l'historique
            messages.append(Message(role="assistant", content=full_response))
            
    finally:
        # Toujours fermer proprement les connexions
        await llm.close()


if __name__ == "__main__":
    # asyncio.run() lance la fonction async main()
    asyncio.run(main())
