set -e
echo "Financial RAG Chatbot — one-command setup"
echo "-----------------------------------------"

if !command -v docker &> /dev/null; then
    echo "Docker is not installed. Install Docker Desktop first: https://docs.docker.com/get-docker/"
    exit 1
fi

mkdir -p data/Documents

echo "Building and starting containers (this pulls the mistral model the first time — may take a few minutes)..."
docker compose up --build