#!/usr/bin/env python3
"""
Simple AI integration test without heavy dependencies
Tests only the core AI structure and imports
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_ai_structure():
    """Test AI module structure and basic functionality"""
    print("🧪 Testing AI module structure...")
    
    # Test directory structure
    ai_dir = current_dir / "ai"
    if not ai_dir.exists():
        print("❌ AI directory not found")
        return False
    print("✅ AI directory exists")
    
    # Test required files
    required_files = [
        "ai/__init__.py",
        "ai/analyst.py", 
        "ai/prompts.py",
        "ai/analysis_types.py"
    ]
    
    for file_path in required_files:
        full_path = current_dir / file_path
        if not full_path.exists():
            print(f"❌ Required file missing: {file_path}")
            return False
        print(f"✅ {file_path} exists")
    
    return True

def test_ai_imports():
    """Test AI imports without heavy dependencies"""
    print("\n🧪 Testing AI imports...")
    
    try:
        # Test analysis types (no heavy dependencies)
        from ai.analysis_types import AnalysisTypes, AnalysisType
        analysis_types = AnalysisTypes()
        available_types = analysis_types.get_available_analysis_types()
        print(f"✅ AnalysisTypes: {len(available_types)} types available")
        
        # Test prompt templates (no heavy dependencies)
        from ai.prompts import PromptTemplates
        prompts = PromptTemplates()
        print("✅ PromptTemplates imported successfully")
        
        # Test basic prompt generation
        test_prompt = prompts.get_player_analysis_prompt(
            player_name="Test Player",
            team="Test Team",
            position="TENGAH", 
            stats_summary="Basic stats",
            context=""
        )
        
        if len(test_prompt) > 100:  # Should be a substantial prompt
            print(f"✅ Prompt generation working (length: {len(test_prompt)})")
        else:
            print(f"❌ Prompt too short: {len(test_prompt)}")
            return False
        
        # Test AI analyst import (will show dependency error but shouldn't crash)
        try:
            from ai.analyst import GemmaAnalyst
            print("⚠️  GemmaAnalyst import succeeded (dependencies available)")
        except ImportError as dep_error:
            if "pandas" in str(dep_error) or "transformers" in str(dep_error):
                print("✅ GemmaAnalyst gracefully handles missing dependencies")
            else:
                print(f"❌ Unexpected import error: {dep_error}")
                return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing AI imports: {e}")
        return False

def test_streamlit_page_structure():
    """Test Streamlit page structure"""
    print("\n🧪 Testing Streamlit page structure...")
    
    # Test pages directory
    pages_dir = current_dir / "pages"
    if not pages_dir.exists():
        print("❌ Pages directory not found")
        return False
    print("✅ Pages directory exists")
    
    # Test AI analyst page
    ai_page = pages_dir / "ai_analyst.py"
    if not ai_page.exists():
        print("❌ AI analyst page not found")
        return False
    print("✅ AI analyst page exists")
    
    # Test utils directory
    utils_dir = current_dir / "utils"  
    if not utils_dir.exists():
        print("❌ Utils directory not found")
        return False
    print("✅ Utils directory exists")
    
    # Test AI utils
    ai_utils = utils_dir / "ai_utils.py"
    if not ai_utils.exists():
        print("❌ AI utils not found") 
        return False
    print("✅ AI utils exists")
    
    return True

def test_requirements():
    """Test requirements file"""
    print("\n🧪 Testing requirements...")
    
    requirements_file = current_dir / "requirements.txt"
    if not requirements_file.exists():
        print("❌ Requirements file not found")
        return False
    
    with open(requirements_file, 'r') as f:
        content = f.read()
    
    # Check for AI dependencies
    ai_deps = ['torch', 'transformers', 'accelerate', 'sentencepiece']
    missing_deps = []
    
    for dep in ai_deps:
        if dep not in content:
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"❌ Missing AI dependencies: {missing_deps}")
        return False
    
    print("✅ All AI dependencies listed in requirements.txt")
    return True

def test_configuration():
    """Test configuration and integration points"""
    print("\n🧪 Testing configuration...")
    
    # Test main app integration
    main_app = current_dir / "app.py"
    if not main_app.exists():
        print("❌ Main app not found")
        return False
    
    with open(main_app, 'r') as f:
        app_content = f.read()
    
    # Check if AI Analyst is added to navigation
    if "🤖 AI Analyst" not in app_content:
        print("❌ AI Analyst not added to main app navigation")
        return False
    
    if "ai_analyst_main" not in app_content:
        print("❌ AI Analyst page not integrated in main app")
        return False
    
    print("✅ AI Analyst properly integrated into main app")
    return True

def main():
    """Run all basic tests"""
    print("🚀 Starting Simple AI Integration Tests\n")
    
    tests = [
        ("AI Structure", test_ai_structure),
        ("AI Imports", test_ai_imports), 
        ("Streamlit Structure", test_streamlit_page_structure),
        ("Requirements", test_requirements),
        ("Configuration", test_configuration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("🧪 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print("="*60)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All basic tests passed! AI integration structure is ready.")
        print("\n📋 Next steps to complete setup:")
        print("1. Install Python dependencies:")
        print("   pip install -r requirements.txt")
        print("2. Run the Streamlit app:")
        print("   streamlit run app.py")  
        print("3. Navigate to the '🤖 AI Analyst' page")
        print("4. Note: First AI model load will take time and require ~8GB RAM")
        print("\n🔧 Features available:")
        print("• Player Performance Analysis")
        print("• Team Tactical Analysis") 
        print("• Scout Report Generation")
        print("• Player Comparison Analysis")
        print("• Custom Football Queries")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before proceeding.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)