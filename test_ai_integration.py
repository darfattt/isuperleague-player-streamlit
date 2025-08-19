#!/usr/bin/env python3
"""
Test script for AI Football Performance Analyst integration
This script tests the core components without requiring Streamlit
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading functionality"""
    print("🧪 Testing data loading...")
    
    try:
        from utils.data_loader import PlayerDataLoader
        
        # Test data loader initialization
        loader = PlayerDataLoader()
        print("✅ PlayerDataLoader initialized successfully")
        
        # Test data loading
        df = loader.load_data()
        print(f"✅ Data loaded successfully: {len(df)} players from {df['Team'].nunique()} teams")
        
        # Test metric categories
        categories = loader.get_metric_categories()
        print(f"✅ Metric categories available: {list(categories.keys())}")
        
        return df, loader
        
    except Exception as e:
        print(f"❌ Error testing data loading: {str(e)}")
        return None, None

def test_ai_components():
    """Test AI components without model loading"""
    print("\n🧪 Testing AI components...")
    
    try:
        from ai.analysis_types import AnalysisTypes, AnalysisType
        from ai.prompts import PromptTemplates
        
        # Test analysis types
        analysis_types = AnalysisTypes()
        available_types = analysis_types.get_available_analysis_types()
        print(f"✅ Analysis types loaded: {len(available_types)} types available")
        
        # Test prompt templates
        prompts = PromptTemplates()
        print("✅ Prompt templates initialized successfully")
        
        # Test specific prompt generation
        test_prompt = prompts.get_player_analysis_prompt(
            player_name="Test Player",
            team="Test Team", 
            position="TENGAH",
            stats_summary="Test stats",
            context="Test context"
        )
        print("✅ Prompt generation working correctly")
        print(f"   Sample prompt length: {len(test_prompt)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing AI components: {str(e)}")
        return False

def test_ai_utils():
    """Test AI utilities"""
    print("\n🧪 Testing AI utilities...")
    
    try:
        from utils.ai_utils import FootballDataProcessor
        
        # Test processor initialization
        processor = FootballDataProcessor()
        print("✅ FootballDataProcessor initialized successfully")
        
        # Test with sample data
        sample_player = pd.Series({
            'Player Name': 'Test Player',
            'Team': 'Test Team',
            'Position': 'TENGAH',
            'Age': 25,
            'Appearances': 10,
            'Goal': 5,
            'Assist': 3,
            'Passing': 100,
            'Tackle': 15,
            'Block': 8
        })
        
        # Test performance score calculation
        score = processor.calculate_performance_score(sample_player)
        print(f"✅ Performance score calculation: {score}")
        
        # Test strengths identification  
        strengths = processor.identify_player_strengths(sample_player)
        print(f"✅ Strengths identification: {len(strengths)} strengths found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing AI utilities: {str(e)}")
        return False

def test_ai_analyst_initialization():
    """Test AI analyst initialization (without model loading)"""
    print("\n🧪 Testing AI analyst initialization...")
    
    try:
        from ai.analyst import GemmaAnalyst
        
        # Initialize analyst (but don't load model)
        analyst = GemmaAnalyst()
        print("✅ GemmaAnalyst initialized successfully")
        
        # Test that components are accessible
        print(f"✅ Model name configured: {analyst.model_name}")
        print(f"✅ Prompt templates available: {analyst.prompt_templates is not None}")
        print(f"✅ Analysis types available: {analyst.analysis_types is not None}")
        
        # Note: We skip actual model loading in this test since it requires significant resources
        print("ℹ️  Model loading skipped in test (requires significant memory)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing AI analyst: {str(e)}")
        return False

def test_integration_with_sample_data():
    """Test integration with actual sample data"""
    print("\n🧪 Testing integration with sample data...")
    
    try:
        # Load real data
        df, loader = test_data_loading()
        if df is None:
            print("❌ Cannot test integration without data")
            return False
        
        # Test with a sample player
        if len(df) > 0:
            sample_player = df.iloc[0]
            print(f"✅ Testing with player: {sample_player.get('Player Name', 'Unknown')}")
            
            # Test AI utilities with real data
            from utils.ai_utils import FootballDataProcessor
            processor = FootballDataProcessor()
            
            score = processor.calculate_performance_score(sample_player)
            strengths = processor.identify_player_strengths(sample_player, top_n=3)
            
            print(f"✅ Performance score: {score}")
            print(f"✅ Top strengths: {[s[0] for s in strengths]}")
            
            # Test prompt generation with real data
            from ai.prompts import PromptTemplates
            prompts = PromptTemplates()
            
            stats_summary = f"""
            Player: {sample_player.get('Player Name', 'Unknown')}
            Team: {sample_player.get('Team', 'Unknown')}
            Position: {sample_player.get('Position', 'Unknown')}
            Goals: {sample_player.get('Goal', 0)}
            Assists: {sample_player.get('Assist', 0)}
            """
            
            prompt = prompts.get_player_analysis_prompt(
                player_name=sample_player.get('Player Name', 'Unknown'),
                team=sample_player.get('Team', 'Unknown'),
                position=sample_player.get('Position', 'Unknown'),
                stats_summary=stats_summary
            )
            
            print(f"✅ Generated analysis prompt ({len(prompt)} chars)")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing integration: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting AI Football Performance Analyst Integration Tests\n")
    
    test_results = []
    
    # Run tests
    df, loader = test_data_loading()
    test_results.append(("Data Loading", df is not None))
    
    ai_components_ok = test_ai_components()
    test_results.append(("AI Components", ai_components_ok))
    
    ai_utils_ok = test_ai_utils()
    test_results.append(("AI Utilities", ai_utils_ok))
    
    ai_analyst_ok = test_ai_analyst_initialization()
    test_results.append(("AI Analyst", ai_analyst_ok))
    
    integration_ok = test_integration_with_sample_data()
    test_results.append(("Integration", integration_ok))
    
    # Summary
    print("\n" + "="*60)
    print("🧪 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print("="*60)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AI integration is ready.")
        print("\n📋 Next steps:")
        print("1. Install AI dependencies: pip install torch transformers accelerate sentencepiece")
        print("2. Run the Streamlit app: streamlit run app.py")
        print("3. Navigate to the '🤖 AI Analyst' page")
        print("4. Note: First AI model load will take several minutes and require significant RAM")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please fix issues before proceeding.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)