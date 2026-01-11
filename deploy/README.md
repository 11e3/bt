# VBO Trading Bot - GCP Deployment Guide

## 🚀 Quick Start

### 1. GCP Instance Setup

```bash
# SSH into your GCP e2-micro instance
gcloud compute ssh YOUR_INSTANCE_NAME --zone=YOUR_ZONE
```

### 2. Deploy the Bot

```bash
# Download and run install script
curl -sSL https://raw.githubusercontent.com/11e3/bt/main/deploy/install.sh | bash
```

Or manually:

```bash
# Clone repository
git clone https://github.com/11e3/bt.git ~/bt
cd ~/bt

# Run install script
chmod +x deploy/install.sh
./deploy/install.sh
```

### 3. Configure API Keys

```bash
# Edit environment file
nano ~/.env

# Add your account keys (you can add multiple accounts):
ACCOUNT_1_NAME=main
ACCOUNT_1_ACCESS_KEY=your_key_here
ACCOUNT_1_SECRET_KEY=your_secret_here

# Optional: Add a second account
ACCOUNT_2_NAME=secondary
ACCOUNT_2_ACCESS_KEY=your_second_key_here
ACCOUNT_2_SECRET_KEY=your_second_secret_here

# Add Telegram credentials
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### 4. Start the Bot

```bash
sudo systemctl start vbo-bot
sudo systemctl status vbo-bot
```

---

## 📋 Commands Reference

| Command | Description |
|---------|-------------|
| `sudo systemctl start vbo-bot` | Start the bot |
| `sudo systemctl stop vbo-bot` | Stop the bot |
| `sudo systemctl restart vbo-bot` | Restart the bot |
| `sudo systemctl status vbo-bot` | Check status |
| `sudo journalctl -u vbo-bot -f` | View live logs |
| `sudo journalctl -u vbo-bot --since today` | Today's logs |

---

## 🔑 Getting API Keys

### Upbit API
1. Go to https://upbit.com/mypage/open_api_management
2. Create new API key
3. Enable: **주문 조회**, **주문**, **출금** (optional)
4. Set IP whitelist to your GCP instance's external IP

### Telegram Bot
1. Message @BotFather on Telegram
2. Send `/newbot` and follow instructions
3. Copy the bot token

### Telegram Chat ID
1. Message your new bot
2. Visit: `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Find `"chat":{"id":XXXXXXX}` - that's your chat ID

---

## ⚙️ Strategy Parameters

Edit `.env` to customize:

```bash
# How many top momentum coins to trade
TOP_N=3

# Momentum lookback period (days)
MOM_LOOKBACK=15

# VBO noise/MA lookback
LOOKBACK=5

# Long MA = LOOKBACK * MULTIPLIER
MULTIPLIER=2

# Price check interval (minutes)
CHECK_INTERVAL_MINUTES=5
```

---

## 📊 Strategy Overview

```
Daily at 00:00 KST:
├── Calculate momentum ranking (20-day returns)
├── Select top N coins
├── Calculate VBO target prices:
│   └── Target = Open + (Prev Range × Noise SMA)
└── Calculate MA filters

Every 5 minutes:
├── Check BUY conditions:
│   ├── In top N momentum?
│   ├── Price ≥ Target (breakout)?
│   ├── Price > Short MA?
│   └── Price > Long MA?
│
└── Check SELL conditions:
    └── Price < Short MA?
```

---

## 🏦 Multiple Account Management

Bot now supports trading with multiple Upbit accounts simultaneously!

### Add Accounts

```bash
# Edit .env file
nano ~/.env

# Add as many accounts as needed:
ACCOUNT_1_NAME=personal
ACCOUNT_1_ACCESS_KEY=key1
ACCOUNT_1_SECRET_KEY=secret1

ACCOUNT_2_NAME=company
ACCOUNT_2_ACCESS_KEY=key2
ACCOUNT_2_SECRET_KEY=secret2

ACCOUNT_3_NAME=fund
ACCOUNT_3_ACCESS_KEY=key3
ACCOUNT_3_SECRET_KEY=secret3
```

### How It Works

- Bot runs **same strategy** for each account independently
- Each account gets its own **Top N momentum ranking**
- Trades are executed **simultaneously** across all accounts
- Telegram notifications include **account name** for each trade

### Example Notification

```
🟢 BUY BTC [personal]
Amount: 3,333,000 KRW
Price: 145,230,000

🟢 BUY BTC [company]
Amount: 5,000,000 KRW
Price: 145,230,000

🟢 BUY BTC [fund]
Amount: 2,000,000 KRW
Price: 145,230,000
```

---

### Bot not starting
```bash
# Check logs for errors
sudo journalctl -u vbo-bot -n 50

# Verify .env file exists
ls -la ~/bt/.env

# Test manually
cd ~/bt
source .venv/bin/activate
python bot.py
```

### API errors
- Check IP whitelist on Upbit
- Verify API key permissions
- Check API key hasn't expired

### Memory issues (e2-micro = 1GB RAM)
```bash
# Add swap if needed
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## ⚠️ Disclaimer

This bot is for educational purposes. Cryptocurrency trading involves significant risk. Use at your own risk.
