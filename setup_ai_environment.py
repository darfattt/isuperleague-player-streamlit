#!/usr/bin/env python3
"""
AI Environment Setup Script for Indonesia Super League Football Analyst
Handles compatible package installation and dependency resolution
"""

import sys
import subprocess
import os
import platform
import venv
from pathlib import Path

def print_header():
    """Print setup script header"""
    print("=" * 70)
    print("🤖 AI Football Analyst Environment Setup")
    print("🏟️ Indonesia Super League Dashboard")
    print("=" * 70)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    python_version = sys.version_info
    min_version = (3, 8)
    recommended_version = (3, 10)
    
    if python_version < min_version:
        print(f"❌ Python {python_version.major}.{python_version.minor} is not supported")
        print(f"   Minimum required: Python {min_version[0]}.{min_version[1]}")
        return False
    
    if python_version < recommended_version:
        print(f"⚠️  Python {python_version.major}.{python_version.minor} detected")
        print(f"   Recommended: Python {recommended_version[0]}.{recommended_version[1]}+ for best compatibility")
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor} - Compatible")
    
    return True

def check_system_requirements():
    """Check system requirements for Windows with graceful degradation"""
    print("\n💻 Checking system requirements...")
    
    # Check RAM with graceful fallback
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"   RAM: {ram_gb:.1f} GB detected")
        
        # Updated for Gemma-3-270M requirements (much lower)
        if ram_gb < 2:
            print("❌ Warning: Less than 2GB RAM detected")
            print("   Even the lightweight AI model may struggle")
            return False
        elif ram_gb < 4:
            print("⚠️  Warning: Less than 4GB RAM detected")
            print("   AI model will work but may be slow")
        elif ram_gb >= 8:
            print("✅ Excellent RAM for AI performance")
        else:
            print("✅ Sufficient RAM for AI model")
            
    except ImportError:
        print("⚠️  Could not import psutil")
        print("   Installing psutil for system monitoring...")
        
        # Try to install psutil
        pip_cmd = get_pip_command()
        if pip_cmd:
            success, _ = run_command_safely([pip_cmd, "install", "psutil"], "install psutil")
            if success:
                print("✅ psutil installed successfully")
                # Retry RAM check
                try:
                    import psutil
                    ram_gb = psutil.virtual_memory().total / (1024**3)
                    print(f"   RAM: {ram_gb:.1f} GB detected")
                except:
                    print("ℹ️  Still cannot detect RAM - assuming sufficient")
            else:
                print("   Failed to install psutil - continuing anyway")
        
        print("ℹ️  Ensure you have at least 2GB RAM available")
    except Exception as e:
        print(f"ℹ️  Could not detect RAM amount: {e}")
        print("   Assuming sufficient memory available")
    
    # Check disk space
    current_dir = Path.cwd()
    try:
        import shutil
        free_gb = shutil.disk_usage(current_dir).free / (1024**3)
        
        print(f"   Free disk space: {free_gb:.1f} GB")
        
        # Updated for Gemma-3-270M (much smaller)
        if free_gb < 2:
            print("❌ Insufficient disk space")
            print("   Need at least 2GB free space for AI models")
            return False
        elif free_gb < 5:
            print("⚠️  Limited disk space")
            print("   Model download may take longer")
        elif free_gb >= 10:
            print("✅ Excellent disk space available")
        else:
            print("✅ Sufficient disk space available")
            
    except Exception:
        print("ℹ️  Could not detect disk space - ensure you have at least 2GB free")
    
    return True

def create_virtual_environment():
    """Create and setup virtual environment for Windows"""
    print("\n🐍 Setting up virtual environment...")
    
    venv_path = Path("ai_env")
    
    if venv_path.exists():
        print(f"ℹ️  Virtual environment already exists at {venv_path}")
        return str(venv_path)
    
    try:
        # Create virtual environment
        print(f"   Creating virtual environment: {venv_path}")
        venv.create(venv_path, with_pip=True)
        print("✅ Virtual environment created successfully")
        return str(venv_path)
        
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        print("   Continuing with system Python...")
        return None


def validate_executable(exe_path):
    """Validate that an executable exists and is accessible"""
    if not exe_path:
        return False
    
    try:
        # Check if it's a full path
        if os.path.isabs(exe_path):
            return os.path.isfile(exe_path) and os.access(exe_path, os.X_OK)
        
        # Check if it's in PATH
        cmd = exe_path.split() if " " in exe_path else [exe_path]
        result = subprocess.run(cmd + ["--version"], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def get_python_command(venv_path=None):
    """Get appropriate python command for Windows environment with validation"""
    if venv_path:
        python_exe = str(Path(venv_path) / "Scripts" / "python.exe")
        if validate_executable(python_exe):
            return python_exe
        else:
            print(f"⚠️  Virtual environment Python not found: {python_exe}")
            print("   Falling back to system Python")
    
    # Try to find system Python
    detected_python = detect_python_executable()
    if detected_python:
        return detected_python
    else:
        print("❌ No Python executable found!")
        print("   Please ensure Python is installed and in PATH")
        return None

def get_pip_command(venv_path=None):
    """Get appropriate pip command for Windows environment with validation"""
    if venv_path:
        pip_exe = str(Path(venv_path) / "Scripts" / "pip.exe")
        if validate_executable(pip_exe):
            return pip_exe
        else:
            print(f"⚠️  Virtual environment pip not found: {pip_exe}")
            print("   Falling back to system pip")
    
    # Try to find system pip
    pip_candidates = ["pip", "pip3", "py -m pip"]
    
    for candidate in pip_candidates:
        if validate_executable(candidate):
            return candidate
    
    print("❌ No pip executable found!")
    print("   Please ensure pip is installed")
    return None

def run_command_safely(cmd, description, timeout=300):
    """Run a command with comprehensive error handling"""
    try:
        print(f"   Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        
        if isinstance(cmd, str):
            cmd = cmd.split()
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
            
        return True, result.stdout
        
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out after {timeout} seconds")
        print(f"   Command: {' '.join(cmd)}")
        return False, "Timeout"
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with return code {e.returncode}")
        print(f"   Command: {' '.join(cmd)}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False, e.stderr
        
    except FileNotFoundError as e:
        print(f"❌ Command not found: {e}")
        print(f"   Command: {' '.join(cmd)}")
        print("   Check that the executable is installed and in PATH")
        return False, str(e)
        
    except Exception as e:
        print(f"❌ Unexpected error running command: {e}")
        print(f"   Command: {' '.join(cmd)}")
        return False, str(e)

def install_dependencies(venv_path=None):
    """Install compatible dependencies for Windows with robust error handling"""
    print("\n📦 Installing compatible dependencies...")
    
    pip_cmd = get_pip_command(venv_path)
    if not pip_cmd:
        print("❌ Cannot install dependencies without pip")
        return False
    
    # First upgrade pip
    print("   Upgrading pip...")
    cmd = [pip_cmd, "install", "--upgrade", "pip"] if isinstance(pip_cmd, str) else pip_cmd.split() + ["install", "--upgrade", "pip"]
    success, output = run_command_safely(cmd, "pip upgrade")
    
    if success:
        print("✅ Pip upgraded successfully")
    else:
        print("⚠️  Pip upgrade failed, continuing with package installation...")
        print("   This might cause version conflicts")
    
    # Install packages in specific order to avoid conflicts
    install_steps = [
        {
            "name": "NumPy (compatible version)",
            "packages": ["numpy>=1.24.0,<2.0.0"]
        },
        {
            "name": "Basic data science packages", 
            "packages": ["pandas>=2.0.0,<2.3.0", "plotly>=5.15.0"]
        },
        {
            "name": "PyTorch ecosystem",
            "packages": ["torch>=2.1.0,<2.5.0", "safetensors>=0.3.1,<0.5.0"]
        },
        {
            "name": "Transformers and tokenization",
            "packages": ["tokenizers>=0.13.0,<0.20.0", "transformers>=4.42.0,<4.46.0"]
        },
        {
            "name": "Hugging Face and acceleration",
            "packages": ["huggingface-hub>=0.16.4,<0.25.0", "accelerate>=0.20.0,<0.35.0"]
        },
        {
            "name": "Additional AI dependencies",
            "packages": ["sentencepiece>=0.1.99,<0.3.0"]
        },
        {
            "name": "Streamlit interface",
            "packages": ["streamlit>=1.28.0"]
        }
    ]
    
    for step in install_steps:
        print(f"\n   Installing {step['name']}...")
        
        if isinstance(pip_cmd, str):
            cmd = [pip_cmd, "install"] + step["packages"]
        else:
            cmd = pip_cmd.split() + ["install"] + step["packages"]
        
        success, output = run_command_safely(cmd, f"install {step['name']}")
        
        if success:
            print(f"✅ {step['name']} installed successfully")
        else:
            print(f"❌ Failed to install {step['name']}")
            print("   This may cause AI functionality to be unavailable")
            
            # For critical packages, return False
            if step['name'] in ["NumPy (compatible version)", "Basic data science packages"]:
                print("   This is a critical dependency - setup cannot continue")
                return False
            else:
                print("   Continuing with remaining packages...")
    
    return True

def verify_installation(venv_path=None):
    """Verify that key packages can be imported with better error handling"""
    print("\n✅ Verifying installation...")
    
    # Get python command using our helper function
    python_cmd = get_python_command(venv_path)
    if not python_cmd:
        print("❌ Cannot verify installation without Python executable")
        return False
    
    test_imports = [
        ("numpy", "NumPy compatibility"),
        ("pandas", "Pandas data processing"),
        ("torch", "PyTorch deep learning"),
        ("transformers", "Transformers NLP"),
        ("streamlit", "Streamlit interface"),
        ("huggingface_hub", "Hugging Face Hub")
    ]
    
    failed_imports = []
    
    for module, description in test_imports:
        import_cmd = [python_cmd, "-c", f"import {module}; print(f'{module} version: {{getattr({module}, '__version__', 'unknown')}}')"]
        success, output = run_command_safely(import_cmd, f"test {module} import", timeout=30)
        
        if success:
            print(f"✅ {description}: {output.strip()}")
        else:
            print(f"❌ {description}: Import failed")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Some imports failed: {', '.join(failed_imports)}")
        print("   AI functionality may be limited")
        return len(failed_imports) <= 2  # Allow some non-critical failures
    
    return True

def create_activation_script(venv_path):
    """Create Windows batch script to activate the environment"""
    if not venv_path:
        return
    
    print("\n📝 Creating activation script...")
    
    # Windows batch script
    batch_content = f'''@echo off
echo Activating AI Football Analyst Environment...
call "{venv_path}\\Scripts\\activate.bat"
echo Environment activated! You can now run:
echo   streamlit run app.py
'''
    with open("activate_ai_env.bat", "w") as f:
        f.write(batch_content)
    print("✅ Created activate_ai_env.bat")

def print_next_steps(venv_path):
    """Print next steps for Windows users"""
    print("\n" + "=" * 70)
    print("🎉 Setup Complete!")
    print("=" * 70)
    
    if venv_path:
        print("\n📋 Next Steps:")
        print("1. Activate the virtual environment:")
        print(f"   .\\activate_ai_env.bat")
        print(f"   # or manually: {venv_path}\\Scripts\\activate.bat")
    
    print("\n2. Run the Streamlit application:")
    print("   streamlit run app.py")
    
    print("\n3. Navigate to '🤖 AI Analyst' in the sidebar")
    
    print("\n🔑 Hugging Face Token Setup:")
    print("• Create token at: https://huggingface.co/settings/tokens")
    print("• Set environment variable: set HF_TOKEN=your_token_here")
    print("• Or enter token in the AI Analyst interface")
    
    print("\n⚠️  Important Notes:")
    print("• First model download will be ~7GB and take several minutes")
    print("• Ensure stable internet connection for initial setup")
    print("• Model loading requires 8GB+ RAM for optimal performance")
    
    print("\n🔧 Troubleshooting:")
    print("• If you encounter import errors, try recreating the environment")
    print("• For memory issues, close other applications before loading AI")
    print("• Check the AI_INTEGRATION_README.md for detailed help")

def detect_python_executable():
    """Detect and validate Python executable on Windows"""
    python_candidates = [
        sys.executable,  # Current Python
        "python",        # Standard python command
        "python3",       # Python 3 specific
        "py",           # Python launcher (Windows)
        "py -3"         # Python launcher for Python 3
    ]
    
    detected_python = None
    
    for candidate in python_candidates:
        try:
            if candidate == sys.executable:
                # Current executable should always work
                detected_python = candidate
                break
            else:
                # Test if command works
                cmd = candidate.split() if " " in candidate else [candidate]
                result = subprocess.run(cmd + ["--version"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    detected_python = candidate
                    break
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    return detected_python

def print_environment_info():
    """Print environment information for Windows with validation"""
    print("\n🔍 Environment Information:")
    print(f"   Platform: {platform.system()}")
    print(f"   Current Python: {sys.executable}")
    
    # Detect available Python
    python_exe = detect_python_executable()
    if python_exe:
        print(f"   Detected Python: {python_exe}")
        try:
            cmd = python_exe.split() if " " in python_exe else [python_exe]
            result = subprocess.run(cmd + ["--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"   Python Version: {result.stdout.strip()}")
        except:
            pass
    else:
        print("   ⚠️  No Python executable detected in PATH")
    
    print(f"   Working Directory: {Path.cwd()}")
    
    # Check PATH
    path_env = os.environ.get('PATH', '')
    print(f"   PATH entries: {len(path_env.split(os.pathsep))} directories")
    
    return python_exe

def main():
    """Main setup function with comprehensive error handling"""
    print_header()
    
    try:
        # Show environment info for debugging
        detected_python = print_environment_info()
        if not detected_python:
            print("\n❌ No Python executable detected!")
            print("🔧 Troubleshooting steps:")
            print("1. Ensure Python is installed")
            print("2. Add Python to PATH environment variable")
            print("3. Try running: py --version")
            print("4. Or reinstall Python with 'Add to PATH' option")
            return False
        
        # Check prerequisites
        if not check_python_version():
            print("\n❌ Python version check failed!")
            print("🔧 Please install Python 3.8 or newer")
            return False
        
        if not check_system_requirements():
            response = input("\n⚠️  System requirements check failed. Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Setup cancelled.")
                return False
    
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during setup initialization: {e}")
        print("🔧 Please check your Python installation and try again")
        return False
    
    try:
        # Setup environment
        use_venv = input("\n🤔 Create virtual environment? (Y/n): ").lower()
        venv_path = None
        
        if use_venv != 'n':
            print("\n🔧 Setting up virtual environment...")
            venv_path = create_virtual_environment()
            if venv_path:
                print(f"✅ Virtual environment created at: {venv_path}")
            else:
                print("⚠️  Virtual environment creation failed, using system Python")
        
        # Install dependencies
        print("\n🔧 Installing dependencies...")
        if not install_dependencies(venv_path):
            print("\n❌ Dependency installation failed!")
            print("🔧 Troubleshooting:")
            print("1. Check internet connection")
            print("2. Try running: pip install --upgrade pip")
            print("3. Check if pip is working: pip --version")
            return False
        
        # Verify installation
        print("\n🔧 Verifying installation...")
        if not verify_installation(venv_path):
            print("\n⚠️  Installation verification had issues!")
            print("   Some AI features may not work properly")
            print("   Basic Streamlit functionality should still work")
        
        # Create helper scripts
        if venv_path:
            try:
                create_activation_script(venv_path)
            except Exception as e:
                print(f"⚠️  Could not create activation script: {e}")
                print("   You can still activate manually")
        
        # Print completion message
        print_next_steps(venv_path)
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        print("\n🔧 Debugging information:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error details: {str(e)}")
        print(f"   Python executable: {sys.executable}")
        print(f"   Working directory: {Path.cwd()}")
        print(f"   Platform: {platform.system()}")
        
        print("\n💡 Try these solutions:")
        print("1. Run as administrator")
        print("2. Check antivirus software isn't blocking")
        print("3. Ensure sufficient disk space")
        print("4. Try: python -m pip install --upgrade pip")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1)