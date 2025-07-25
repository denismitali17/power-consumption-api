set -e 
echo "Starting build process..."
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
echo "Build complete!"