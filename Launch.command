#!/bin/bash

# Fast Flux Studio - Mac Launcher
# Double-click this file to start the app!

cd "$(dirname "$0")"

echo "============================================"
echo "        Fast Flux Studio for Mac"
echo "============================================"
echo ""

# Check Python version
PYTHON_CMD=""
for cmd in python3.11 python3.10 python3; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd -c 'import sys; print(sys.version_info.minor)')
        if [ "$version" -ge 10 ]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Python 3.10+ is required but not found."
    echo ""

    # Check if Homebrew is installed
    if command -v brew &> /dev/null; then
        echo "Homebrew detected! Would you like to install Python 3.11? (y/n)"
        read -p "> " install_python
        if [ "$install_python" = "y" ] || [ "$install_python" = "Y" ]; then
            echo ""
            echo "Installing Python 3.11..."
            brew install python@3.11
            PYTHON_CMD="python3.11"
            echo ""
            echo "Python 3.11 installed successfully!"
        else
            echo "Please install Python 3.10+ manually and try again."
            read -p "Press Enter to exit..."
            exit 1
        fi
    else
        echo "Would you like to install Homebrew and Python? (y/n)"
        read -p "> " install_brew
        if [ "$install_brew" = "y" ] || [ "$install_brew" = "Y" ]; then
            echo ""
            echo "Installing Homebrew (you may need to enter your password)..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

            # Add Homebrew to PATH for this session
            eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || eval "$(/usr/local/bin/brew shellenv)" 2>/dev/null

            echo ""
            echo "Installing Python 3.11..."
            brew install python@3.11
            PYTHON_CMD="python3.11"
            echo ""
            echo "Installation complete!"
        else
            echo ""
            echo "To install manually:"
            echo "  1. Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            echo "  2. Run: brew install python@3.11"
            read -p "Press Enter to exit..."
            exit 1
        fi
    fi
fi

echo "Using: $PYTHON_CMD"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "First time setup - creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/.installed" ]; then
    echo ""
    echo "Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch venv/.installed
    echo ""
    echo "Installation complete!"
fi

echo ""
echo "Starting Gradio UI..."
echo "Opening browser to http://localhost:7860"
echo ""
echo "(Press Ctrl+C to stop the server)"
echo ""

# Open browser after a short delay
(sleep 3 && open http://localhost:7860) &

# Run the app
python app.py
