#!/bin/bash
#
# 🚀 AI Desktop Companion Launcher
# Double-cliquez sur ce fichier pour lancer l'assistant !
#
# Ce script:
# 1. Vérifie que le serveur LLM (llama.cpp) tourne
# 2. Lance le backend FastAPI
# 3. Lance l'interface desktop avec Live2D
#

cd "$(dirname "$0")"

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════╗"
echo "║       🎭 AI Desktop Companion - March 7th 🎭          ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Vérifier si le serveur LLM tourne
echo -e "${YELLOW}[1/3]${NC} Vérification du serveur LLM..."
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Serveur LLM en cours d'exécution"
else
    echo -e "  ${RED}✗${NC} Serveur LLM non détecté"
    echo -e "  ${YELLOW}→${NC} Démarrage automatique..."
    ./scripts/start_llm_server.sh --daemon
    sleep 3
    
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Serveur LLM démarré !"
    else
        echo -e "  ${RED}⚠${NC} Impossible de démarrer le serveur LLM"
        echo "     Vérifiez les logs ou lancez manuellement:"
        echo "     ./scripts/start_llm_server.sh"
        read -p "Appuyez sur Entrée pour continuer quand même..."
    fi
fi

# Activer l'environnement virtuel et lancer
echo -e "${YELLOW}[2/3]${NC} Activation de l'environnement Python..."
source venv/bin/activate

echo -e "${YELLOW}[3/3]${NC} Lancement de l'assistant..."
echo ""
echo -e "${GREEN}💡 Conseils:${NC}"
echo "   • Parlez dans votre micro pour interagir"
echo "   • Tapez du texte dans la bulle de chat"
echo "   • Fermez la fenêtre pour quitter"
echo ""

# Lancer avec --with-backend pour démarrer backend + frontend
python desktop.py --with-backend
