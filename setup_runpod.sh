#!/bin/bash
# Mochan RunPod Setup Script

echo "🚀 Starting RunPod Setup..."

# 1. Update system and install ffmpeg (needed for audio)
apt-get update && apt-get install -y ffmpeg wget

# 2. Install Python dependencies
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # Optional: speeds up Llama models on CUDA

# 3. Install Cloudflare Tunnel (cloudflared)
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i cloudflared-linux-amd64.deb

echo "✅ Environment Ready!"
echo "-------------------------------------------------------"
echo "STEPS TO START:"
echo "1. Open a SECOND terminal window."
echo "2. Run: cloudflared tunnel --url http://localhost:8765"
echo "3. Copy the 'https://...' URL it gives you."
echo "4. Update your .env file:"
echo "   - Change 'https://' to 'wss://'"
echo "   - Set WEBSOCKET_PUBLIC_URL=wss://your-new-url.trycloudflare.com"
echo "5. Run the assistant: python3 run_mvp.py"
echo "-------------------------------------------------------"
