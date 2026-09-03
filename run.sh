#!/bin/bash

# Quick Setup Script for Anomaly Detection System
# Run this to set up everything automatically

echo "================================================"
echo "🚀 Anomaly Detection System - Quick Setup"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found${NC}"
echo ""

# Create virtual environment
echo "🌍 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p models
mkdir -p outputs
mkdir -p data
mkdir -p logs
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Check for trained model
echo "🔍 Checking for trained model..."
if [ ! -f "models/log_transformer.keras" ]; then
    echo -e "${YELLOW}⚠ No trained model found${NC}"
    echo "   You'll need to train a model first using:"
    echo "   python new.py --train_csv data/your_data.csv"
else
    echo -e "${GREEN}✓ Trained model found${NC}"
fi
echo ""

# Create sample data if needed
echo "📊 Checking for data..."
if [ ! -f "data/train_data.csv" ]; then
    echo -e "${YELLOW}⚠ No training data found${NC}"
    echo "   Please place your training data in: data/train_data.csv"
    echo ""
    echo "   Expected format:"
    echo "   timestamp,entity_id,event_type,dst_id,bytes,Label"
    echo ""
else
    echo -e "${GREEN}✓ Training data found${NC}"
fi
echo ""

# Test imports
echo "🧪 Testing imports..."
python3 -c "
import tensorflow as tf
import flask
import pandas as pd
import numpy as np
print('✓ All imports successful')
print(f'✓ TensorFlow version: {tf.__version__}')
" 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Import test passed${NC}"
else
    echo -e "${RED}✗ Import test failed${NC}"
    exit 1
fi
echo ""

# Summary
echo "================================================"
echo "✨ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Train your model (if you haven't already):"
echo "   ${GREEN}python new.py --train_csv data/your_data.csv${NC}"
echo ""
echo "2. Start the API server:"
echo "   ${GREEN}python backend.py${NC}"
echo ""
echo "3. Open your browser:"
echo "   ${GREEN}http://localhost:5000${NC}"
echo ""
echo "================================================"
echo ""
echo "For detailed documentation, see README.md"
echo ""
echo "Happy anomaly hunting! 🔍✨"
echo ""
