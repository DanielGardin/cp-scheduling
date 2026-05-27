find . -name "*.so" -type f -delete
rm -rf build/

# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

pip uninstall -y -q cpscheduler