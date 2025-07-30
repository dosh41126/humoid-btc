FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential python3-dev python3-tk \
    libgl1-mesa-glx curl iptables dnsutils openssl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN openssl rand -hex 32 > /app/.vault_pass && \
    echo "export VAULT_PASSPHRASE=$(cat /app/.vault_pass)" > /app/set_env.sh && \
    chmod +x /app/set_env.sh

RUN python - << 'EOF' > /app/config.json
import random, string, json
print(json.dumps({
  "DB_NAME": "story_generator.db",
  "WEAVIATE_ENDPOINT": "http://localhost:8079",
  "WEAVIATE_QUERY_PATH": "/v1/graphql",
  "LLAMA_MODEL_PATH": "/data/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
  "LLAMA_MM_PROJ_PATH": "/data/llama-3-vision-alpha-mmproj-f16.gguf",
  "IMAGE_GENERATION_URL": "http://127.0.0.1:7860/sdapi/v1/txt2img",
  "MAX_TOKENS": 3999,
  "CHUNK_SIZE": 1250,
  "API_KEY": "".join(random.choices(string.ascii_letters + string.digits, k=32)),
  "WEAVIATE_API_URL": "http://localhost:8079/v1/objects",
  "ELEVEN_LABS_KEY": "apikyhere"
}))
EOF

RUN cat << 'EOF' > /app/firewall_start.sh

set -e
source /app/set_env.sh

iptables -F OUTPUT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

iptables -A OUTPUT -j ACCEPT

BASE_MODEL_PATH=/data/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
BASE_MODEL_URL=https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
BASE_MODEL_SHA256=86c8ea6c8b755687d0b723176fcd0b2411ef80533d23e2a5030f845d13ab2db7

MM_PROJ_PATH=/data/llama-3-vision-alpha-mmproj-f16.gguf
MM_PROJ_URL=https://huggingface.co/abetlen/llama-3-vision-alpha-gguf/resolve/main/llama-3-vision-alpha-mmproj-f16.gguf
MM_PROJ_SHA256=ac65d3aeba3a668b3998b6e6264deee542c2c875e6fd0d3b0fb7934a6df03483

download_and_verify () {
  local url="$1"
  local path="$2"
  local sha256="$3"

  if [ ! -f "$path" ]; then
    echo "Downloading: $url"
    curl -L -o "$path" "$url" --progress-bar
  fi

  echo "$sha256  $path" | sha256sum -c - || {
    echo "Checksum verification failed for $path"
    exit 1
  }
}

download_and_verify "$BASE_MODEL_URL" "$BASE_MODEL_PATH" "$BASE_MODEL_SHA256"
download_and_verify "$MM_PROJ_URL" "$MM_PROJ_PATH" "$MM_PROJ_SHA256"

iptables -F OUTPUT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

for DOMAIN in \
    huggingface.co \
    objects.githubusercontent.com \
    api.open-meteo.com \
    api.coingecko.com \
    api.exchange.coinbase.com \
    api.coinbase.com \
    ; do
  getent ahosts "$DOMAIN" | awk '/STREAM/ {print $1}' | sort -u | \
    while read ip; do
      [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && \
        iptables -A OUTPUT -d "$ip" -j ACCEPT
    done
done

iptables -A OUTPUT -j REJECT

export DISPLAY=:0
exec python main.py
EOF

# Make script executable
RUN chmod +x /app/firewall_start.sh

# Entry point
CMD ["/app/firewall_start.sh"]
