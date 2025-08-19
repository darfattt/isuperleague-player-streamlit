#!/usr/bin/env python3
"""
Validation script for the Player Analytics Streamlit app
Run this to check if the app is ready to run
"""

import sys
from pathlib import Path

def validate_app():
    """Validate app structure, syntax, and data"""
    print("🔍 Validating Indonesia Super League Player Analytics App...")
    print("=" * 60)
    
    # Check file structure
    print("\n📁 File Structure Check:")
    required_files = [
        'app.py',
        'requirements.txt', 
        'README.md',
        'utils/data_loader.py',
        'utils/visualization.py',
        'data/players_statistics.csv'
    ]
    
    all_files_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Some required files are missing!")
        return False
    
    # Check Python syntax
    print("\n🐍 Python Syntax Check:")
    python_files = ['app.py', 'utils/data_loader.py', 'utils/visualization.py']
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                compile(content, file_path, 'exec')
            print(f"   ✅ {file_path}")
        except SyntaxError as e:
            print(f"   ❌ {file_path} - Syntax Error: {e}")
            return False
        except Exception as e:
            print(f"   ⚠️  {file_path} - Warning: {e}")
    
    # Check data file
    print("\n📊 Data File Check:")
    try:
        with open('data/players_statistics.csv', 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            
            # Check required columns
            required_columns = ['Name', 'Player Name', 'Team', 'Position', 'Age', 'Appearances']
            header_columns = header.split(',')
            
            missing_columns = [col for col in required_columns if col not in header_columns]
            if missing_columns:
                print(f"   ❌ Missing required columns: {missing_columns}")
                return False
            
            # Count data rows
            data_rows = sum(1 for line in f)
            print(f"   ✅ CSV header looks good")
            print(f"   ✅ Found {data_rows} data rows")
            
            # Check for positions including P. GAWANG
            with open('data/players_statistics.csv', 'r', encoding='utf-8') as f2:
                content = f2.read()
                if 'P. GAWANG' in content:
                    print(f"   ✅ Goalkeeper position (P. GAWANG) detected")
                
    except Exception as e:
        print(f"   ❌ Error reading CSV: {e}")
        return False
    
    # Check app configuration
    print("\n⚙️ App Configuration Check:")
    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            app_content = f.read()
            
        # Check for key functions
        key_functions = [
            'def load_data():',
            'def main():',
            'def show_stats_dashboard(',
            'def show_metric_performance('
        ]
        
        for func in key_functions:
            if func in app_content:
                print(f"   ✅ {func.replace('def ', '').replace('(', ' function found')}")
            else:
                print(f"   ❌ {func} - MISSING")
                return False
        
        # Check for position handling
        if 'P. GAWANG' in app_content:
            print(f"   ✅ Goalkeeper position handling")
        else:
            print(f"   ❌ Missing P. GAWANG position handling")
            return False
            
        # Check for error handling
        if 'try:' in app_content and 'except' in app_content:
            print(f"   ✅ Error handling implemented")
        else:
            print(f"   ⚠️  Limited error handling")
        
    except Exception as e:
        print(f"   ❌ Error checking app configuration: {e}")
        return False
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ VALIDATION PASSED - App is ready to run!")
    print("\nTo start the app:")
    print("   pip install -r requirements.txt")
    print("   streamlit run app.py")
    print("\nExpected features:")
    print("   • Player statistics dashboard")
    print("   • Global filters (Team, Position, Age, Appearances)")
    print("   • Support for all positions including goalkeepers")
    print("   • Proper handling of negative metrics")
    print("   • Error handling and data validation")
    
    return True

if __name__ == "__main__":
    success = validate_app()
    sys.exit(0 if success else 1)