#!/bin/bash
# VBO Bot Deployment Script for GCP e2-micro
# Run this script on your GCP instance

set -e

echo "=========================================="
echo "VBO Trading Bot - GCP Deployment"
echo "=========================================="

# Variables
APP_DIR="/home/ubuntu/bt"
VENV_DIR="$APP_DIR/.venv"
SERVICE_NAME="vbo-bot"

# Update system
echo "[1/7] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+
echo "[2/7] Installing Python..."
sudo apt install -y python3 python3-pip python3-venv git

# Clone or update repository
echo "[3/7] Setting up application..."
if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR"
    git pull
else
    git clone https://github.com/11e3/bt.git "$APP_DIR"
    cd "$APP_DIR"
fi

# Create virtual environment
echo "[4/7] Creating virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "[5/7] Installing dependencies..."
pip install --upgrade pip
pip install pyupbit pandas httpx pydantic pydantic-settings PyJWT

# Setup environment file
echo "[6/7] Setting up environment..."
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo "⚠️  Please edit $APP_DIR/.env with your API keys!"
    echo "   nano $APP_DIR/.env"
fi

# Install systemd service
echo "[7/7] Installing systemd service..."
sudo cp "$APP_DIR/deploy/vbo-bot.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit your API keys:"
echo "   nano $APP_DIR/.env"
echo ""
echo "2. Start the bot:"
echo "   sudo systemctl start $SERVICE_NAME"
echo ""
echo "3. Check status:"
echo "   sudo systemctl status $SERVICE_NAME"
echo ""
echo "4. View logs:"
echo "   sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "5. Stop the bot:"
echo "   sudo systemctl stop $SERVICE_NAME"
echo ""
