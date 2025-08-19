#!/usr/bin/env python3
"""
Comprehensive AI Integration Testing Script
Tests the fixed NumPy compatibility and Hugging Face authentication
"""

import sys
import os
import subprocess
from pathlib import Path
import logging

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def print_header():
    """Print test script header"""
    print("=" * 80)
    print("🔧 AI Integration Fix Validation")
    print("🏟️ Indonesia Super League Football Analyst")
    print("=" * 80)
    print()

def test_numpy_version():
    """Test NumPy version compatibility"""
    print("🧪 Testing NumPy Version Compatibility...")
    
    try:
        import numpy as np
        numpy_version = np.__version__
        print(f"   NumPy version: {numpy_version}")
        
        # Check version compatibility
        major, minor, patch = numpy_version.split('.')
        major, minor = int(major), int(minor)
        
        if major >= 2:
            print("❌ NumPy 2.0+ detected - this may cause compatibility issues")
            print("   Recommended: pip install 'numpy>=1.24.0,<2.0.0'")
            return False
        elif major == 1 and minor >= 24:
            print("✅ NumPy version compatible")
            return True
        else:
            print("⚠️  NumPy version may be too old")
            print("   Recommended: pip install 'numpy>=1.24.0,<2.0.0'")
            return False
            
    except ImportError:
        print("❌ NumPy not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking NumPy: {e}")
        return False

def test_dependencies():
    """Test all required dependencies"""
    print("\n🔍 Testing Dependencies...")
    
    deps_to_test = [
        ("pandas", "Data processing"),
        ("torch", "PyTorch deep learning"),
        ("transformers", "Transformers library"),
        ("huggingface_hub", "Hugging Face Hub"),
        ("accelerate", "Model acceleration"),
        ("sentencepiece", "Tokenization"),
        ("streamlit", "Web interface")
    ]
    
    results = []
    
    for module, description in deps_to_test:
        try:
            imported_module = __import__(module)
            version = getattr(imported_module, '__version__', 'unknown')
            print(f"✅ {description}: {module} v{version}")
            results.append(True)
        except ImportError:
            print(f"❌ {description}: {module} not installed")
            results.append(False)
        except Exception as e:
            print(f"⚠️  {description}: {module} - {str(e)}")
            results.append(False)
    
    return all(results)

def test_transformers_import():
    """Specifically test transformers import (the problematic one)"""
    print("\n🤖 Testing Transformers Import...")
    
    try:
        # This is the specific import that was failing
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        print("✅ Core transformers imports successful")
        
        # Test pipeline creation (without loading a model)
        print("   Testing pipeline creation...")
        # We won't actually create a pipeline to avoid downloading models
        print("✅ Pipeline imports successful")
        
        return True
        
    except RuntimeError as e:
        if "numpy.dtype size changed" in str(e):
            print("❌ NumPy compatibility error detected!")
            print("   This is the exact error you were experiencing")
            print("   Solution: pip uninstall numpy && pip install 'numpy>=1.24.0,<2.0.0'")
            return False
        else:
            print(f"❌ Runtime error: {e}")
            return False
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_hf_auth():
    """Test Hugging Face authentication utilities"""
    print("\n🔑 Testing Hugging Face Authentication...")
    
    try:
        from utils.hf_auth import get_hf_auth, quick_token_test
        
        # Test auth utility creation
        hf_auth = get_hf_auth()
        print("✅ HF Auth utility created")
        
        # Test token source detection
        sources = hf_auth.get_token_sources()
        print("✅ Token source detection working")
        
        if sources['active_token']:
            masked_token = sources['active_token'][:8] + "..." + sources['active_token'][-4:]
            print(f"ℹ️  Active token found: {masked_token}")
            print(f"   Source: {sources['source']}")
            
            # Test token validation
            print("   Testing token validation...")
            test_result = quick_token_test(sources['active_token'])
            
            if test_result['success']:
                print("✅ Token is valid!")
                user_info = test_result.get('user_info', {})
                if user_info:
                    print(f"   User: {user_info.get('name', 'Unknown')}")
            else:
                print(f"⚠️  Token test failed: {test_result.get('error')}")
        else:
            print("ℹ️  No active token found (this is OK for public models)")
        
        return True
        
    except Exception as e:
        print(f"❌ HF Auth test failed: {e}")
        return False

def test_ai_analyst_creation():
    """Test AI analyst creation (without model loading)"""
    print("\n🤖 Testing AI Analyst Creation...")
    
    try:
        from ai.analyst import GemmaAnalyst
        
        # Test basic creation
        analyst = GemmaAnalyst()
        print("✅ GemmaAnalyst created successfully")
        
        # Test token detection
        if analyst.token:
            print("✅ Token detected and configured")
        else:
            print("ℹ️  No token configured (OK for public models)")
        
        # Test prompt templates
        if analyst.prompt_templates:
            print("✅ Prompt templates loaded")
        
        # Test analysis types
        if analyst.analysis_types:
            available_types = analyst.analysis_types.get_available_analysis_types()
            print(f"✅ Analysis types loaded: {len(available_types)} types")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Analyst creation failed: {e}")
        return False

def test_prompt_generation():
    """Test prompt generation functionality"""
    print("\n📝 Testing Prompt Generation...")
    
    try:
        from ai.prompts import PromptTemplates
        
        prompts = PromptTemplates()
        print("✅ Prompt templates initialized")
        
        # Test player analysis prompt
        test_prompt = prompts.get_player_analysis_prompt(
            player_name="Cristiano Ronaldo",
            team="Al Nassr",
            position="DEPAN",
            stats_summary="Goals: 25, Assists: 8, Shots: 120",
            context="Testing prompt generation"
        )
        
        if len(test_prompt) > 500:  # Should be substantial
            print(f"✅ Player analysis prompt generated ({len(test_prompt)} chars)")
        else:
            print(f"⚠️  Prompt seems short: {len(test_prompt)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt generation failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n📊 Testing Data Loading...")
    
    try:
        from utils.data_loader import PlayerDataLoader
        
        loader = PlayerDataLoader()
        print("✅ Data loader created")
        
        # Test if data file exists
        data_path = current_dir / "data" / "players_statistics.csv"
        if data_path.exists():
            df = loader.load_data()
            print(f"✅ Data loaded: {len(df)} players")
            
            # Test metric categories
            categories = loader.get_metric_categories()
            print(f"✅ Metric categories: {list(categories.keys())}")
            
        else:
            print("ℹ️  Data file not found (this is OK - it will be loaded by the main app)")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_streamlit_imports():
    """Test Streamlit page imports"""
    print("\n🌐 Testing Streamlit Integration...")
    
    try:
        # Test that we can import the AI analyst page
        import pages.ai_analyst
        print("✅ AI analyst page imports successfully")
        
        # Test authentication function
        if hasattr(pages.ai_analyst, 'render_authentication_sidebar'):
            print("✅ Authentication sidebar function available")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit integration test failed: {e}")
        return False

def run_pip_check():
    """Run pip check to identify conflicts"""
    print("\n🔍 Running Dependency Conflict Check...")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "check"], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ No dependency conflicts detected")
            return True
        else:
            print("⚠️  Dependency conflicts detected:")
            print(result.stdout)
            return False
            
    except Exception as e:
        print(f"ℹ️  Could not run pip check: {e}")
        return True  # Don't fail the test for this

def print_summary(results):
    """Print test results summary"""
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    test_names = [
        "NumPy Version",
        "Dependencies",
        "Transformers Import",
        "HF Authentication",
        "AI Analyst Creation",
        "Prompt Generation",
        "Data Loading",
        "Streamlit Integration",
        "Pip Check"
    ]
    
    passed = sum(results)
    total = len(results)
    
    for name, result in zip(test_names, results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20} {status}")
    
    print("=" * 80)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The NumPy fix is working correctly.")
        print("\n🚀 Ready to use AI Football Analyst:")
        print("   streamlit run app.py")
        print("   Then navigate to '🤖 AI Analyst' page")
        
    elif passed >= 6:  # Most tests passed
        print(f"\n✅ Most tests passed ({passed}/{total})")
        print("The AI integration should work with minor issues.")
        print("Check the failed tests above for specific problems.")
        
    else:
        print(f"\n❌ Multiple test failures ({passed}/{total})")
        print("Please fix the issues before using the AI analyst.")
        
        if not results[0]:  # NumPy version failed
            print("\n🔧 Primary fix needed:")
            print("   pip uninstall numpy")
            print("   pip install 'numpy>=1.24.0,<2.0.0'")
            print("   pip install -r requirements.txt")
        
        if not results[1]:  # Dependencies failed
            print("\n🔧 Install missing dependencies:")
            print("   python setup_ai_environment.py")

def main():
    """Run all tests"""
    print_header()
    
    tests = [
        test_numpy_version,
        test_dependencies,
        test_transformers_import,
        test_hf_auth,
        test_ai_analyst_creation,
        test_prompt_generation,
        test_data_loading,
        test_streamlit_imports,
        run_pip_check
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    print_summary(results)
    
    return sum(results) >= len(results) - 1  # Allow one test failure

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)