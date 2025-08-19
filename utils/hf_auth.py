"""
Hugging Face Authentication Utilities
Handles token validation, storage, and authentication testing
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Setup logging
logger = logging.getLogger(__name__)

class HuggingFaceAuth:
    """
    Utility class for managing Hugging Face authentication
    """
    
    def __init__(self):
        self.config_dir = Path.home() / ".huggingface"
        self.token_file = self.config_dir / "token"
        self.config_file = self.config_dir / "config.json"
    
    def get_token_sources(self) -> Dict[str, Any]:
        """
        Get all possible sources for Hugging Face tokens
        
        Returns:
            Dictionary with token source information
        """
        sources = {
            'environment': {
                'HF_TOKEN': os.getenv('HF_TOKEN'),
                'HUGGINGFACE_HUB_TOKEN': os.getenv('HUGGINGFACE_HUB_TOKEN'),
            },
            'config_file': None,
            'token_file': None,
            'active_token': None,
            'source': None
        }
        
        # Check config file
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    sources['config_file'] = config.get('token')
            except Exception as e:
                logger.warning(f"Could not read config file: {e}")
        
        # Check token file
        if self.token_file.exists():
            try:
                with open(self.token_file, 'r') as f:
                    sources['token_file'] = f.read().strip()
            except Exception as e:
                logger.warning(f"Could not read token file: {e}")
        
        # Determine active token (priority order)
        if sources['environment']['HF_TOKEN']:
            sources['active_token'] = sources['environment']['HF_TOKEN']
            sources['source'] = 'HF_TOKEN environment variable'
        elif sources['environment']['HUGGINGFACE_HUB_TOKEN']:
            sources['active_token'] = sources['environment']['HUGGINGFACE_HUB_TOKEN']
            sources['source'] = 'HUGGINGFACE_HUB_TOKEN environment variable'
        elif sources['config_file']:
            sources['active_token'] = sources['config_file']
            sources['source'] = 'config.json file'
        elif sources['token_file']:
            sources['active_token'] = sources['token_file']
            sources['source'] = 'token file'
        
        return sources
    
    def validate_token_format(self, token: str) -> Dict[str, Any]:
        """
        Validate Hugging Face token format
        
        Args:
            token: Token to validate
            
        Returns:
            Validation result dictionary
        """
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        if not token:
            result['errors'].append("Token is empty")
            return result
        
        if not isinstance(token, str):
            result['errors'].append("Token must be a string")
            return result
        
        # Basic format checks
        token = token.strip()
        result['info']['length'] = len(token)
        result['info']['starts_with_hf'] = token.startswith('hf_')
        
        if len(token) < 20:
            result['errors'].append("Token too short (minimum 20 characters)")
        
        if not token.startswith('hf_'):
            result['warnings'].append("Token doesn't start with 'hf_' (may be valid but unusual)")
        
        # Character validation
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')
        invalid_chars = set(token) - allowed_chars
        
        if invalid_chars:
            result['errors'].append(f"Invalid characters found: {', '.join(invalid_chars)}")
        
        result['valid'] = len(result['errors']) == 0
        return result
    
    def test_token_connection(self, token: str) -> Dict[str, Any]:
        """
        Test token connection to Hugging Face Hub
        
        Args:
            token: Token to test
            
        Returns:
            Connection test result
        """
        result = {
            'success': False,
            'user_info': None,
            'error': None,
            'permissions': []
        }
        
        try:
            from huggingface_hub import whoami
            
            # Test basic authentication
            user_info = whoami(token=token)
            result['success'] = True
            result['user_info'] = user_info
            
            # Extract user details
            if isinstance(user_info, dict):
                result['user_name'] = user_info.get('name', 'Unknown')
                result['user_type'] = user_info.get('type', 'unknown')
                result['user_id'] = user_info.get('id', 'unknown')
            
            logger.info(f"Token authentication successful for user: {result.get('user_name', 'Unknown')}")
            
        except ImportError:
            result['error'] = "huggingface_hub library not available"
            logger.error("huggingface_hub not installed for token testing")
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Token authentication failed: {e}")
        
        return result
    
    def save_token(self, token: str, method: str = 'file') -> Dict[str, Any]:
        """
        Save token using specified method
        
        Args:
            token: Token to save
            method: Save method ('file', 'config', 'env_instruction')
            
        Returns:
            Save operation result
        """
        result = {
            'success': False,
            'method': method,
            'path': None,
            'error': None,
            'instructions': []
        }
        
        # Validate token first
        validation = self.validate_token_format(token)
        if not validation['valid']:
            result['error'] = f"Invalid token format: {', '.join(validation['errors'])}"
            return result
        
        try:
            # Create config directory if it doesn't exist
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            if method == 'file':
                # Save to token file
                with open(self.token_file, 'w') as f:
                    f.write(token.strip())
                
                # Set restrictive permissions (Unix-like systems only)
                try:
                    os.chmod(self.token_file, 0o600)
                except (OSError, AttributeError):
                    pass  # Windows or other OS
                
                result['success'] = True
                result['path'] = str(self.token_file)
                result['instructions'].append(f"Token saved to {self.token_file}")
                
            elif method == 'config':
                # Save to config.json
                config = {}
                if self.config_file.exists():
                    try:
                        with open(self.config_file, 'r') as f:
                            config = json.load(f)
                    except Exception:
                        pass  # Start with empty config if file is corrupted
                
                config['token'] = token.strip()
                
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
                result['success'] = True
                result['path'] = str(self.config_file)
                result['instructions'].append(f"Token saved to {self.config_file}")
                
            elif method == 'env_instruction':
                # Provide environment variable instructions
                result['success'] = True
                result['instructions'] = [
                    "To set as environment variable:",
                    f"export HF_TOKEN='{token}'",
                    "Or add to your shell profile (.bashrc, .zshrc, etc.)",
                    "For Windows Command Prompt:",
                    f"set HF_TOKEN={token}",
                    "For Windows PowerShell:",
                    f"$env:HF_TOKEN='{token}'"
                ]
            
            else:
                result['error'] = f"Unknown save method: {method}"
        
        except Exception as e:
            result['error'] = f"Failed to save token: {str(e)}"
            logger.error(f"Token save failed: {e}")
        
        return result
    
    def remove_stored_tokens(self) -> Dict[str, Any]:
        """
        Remove all stored tokens from files
        
        Returns:
            Removal operation result
        """
        result = {
            'success': False,
            'removed_files': [],
            'errors': []
        }
        
        files_to_remove = [self.token_file, self.config_file]
        
        for file_path in files_to_remove:
            if file_path.exists():
                try:
                    if file_path == self.config_file:
                        # For config file, just remove the token key
                        with open(file_path, 'r') as f:
                            config = json.load(f)
                        
                        if 'token' in config:
                            del config['token']
                            
                            if config:  # If there's other config, save it back
                                with open(file_path, 'w') as f:
                                    json.dump(config, f, indent=2)
                                result['removed_files'].append(f"{file_path} (token removed)")
                            else:  # If empty, remove the file
                                file_path.unlink()
                                result['removed_files'].append(str(file_path))
                    else:
                        # For token file, remove entirely
                        file_path.unlink()
                        result['removed_files'].append(str(file_path))
                        
                except Exception as e:
                    result['errors'].append(f"Failed to remove {file_path}: {str(e)}")
        
        result['success'] = len(result['errors']) == 0
        return result
    
    def get_setup_instructions(self) -> List[str]:
        """
        Get step-by-step token setup instructions
        
        Returns:
            List of instruction strings
        """
        return [
            "🔑 Hugging Face Token Setup Instructions:",
            "",
            "1. Create a Hugging Face account:",
            "   https://huggingface.co/join",
            "",
            "2. Generate an access token:",
            "   • Go to: https://huggingface.co/settings/tokens",
            "   • Click 'New token'",
            "   • Choose 'read' permission (sufficient for model access)",
            "   • Copy the generated token (starts with 'hf_')",
            "",
            "3. Configure the token (choose one method):",
            "",
            "   Method A - Environment Variable (Recommended):",
            "   export HF_TOKEN='your_token_here'",
            "",
            "   Method B - Use this interface:",
            "   • Enter token in the 'Authentication' section",
            "   • Click 'Test & Save Token'",
            "",
            "   Method C - Manual file setup:",
            f"   • Create file: {self.token_file}",
            "   • Add your token to the file",
            "",
            "4. Verify token works:",
            "   • Use 'Test Token' button in this interface",
            "   • Should show your username if successful",
            "",
            "⚠️  Security Notes:",
            "• Keep your token secret - don't share it",
            "• Don't commit tokens to version control",
            "• You can revoke/regenerate tokens if compromised"
        ]
    
    def get_troubleshooting_guide(self) -> List[str]:
        """
        Get troubleshooting guide for token issues
        
        Returns:
            List of troubleshooting steps
        """
        return [
            "🔧 Token Troubleshooting Guide:",
            "",
            "Common Issues & Solutions:",
            "",
            "1. 'Token format invalid' error:",
            "   • Ensure token starts with 'hf_'",
            "   • Check for extra spaces or characters",
            "   • Token should be at least 20 characters",
            "",
            "2. 'Authentication failed' error:",
            "   • Verify token at: https://huggingface.co/settings/tokens",
            "   • Check if token has been revoked",
            "   • Try generating a new token",
            "",
            "3. 'Permission denied' error:",
            "   • Some models require special access",
            "   • Check model page for access requirements",
            "   • Ensure token has appropriate permissions",
            "",
            "4. 'Network connection' error:",
            "   • Check internet connection",
            "   • Verify Hugging Face Hub is accessible",
            "   • Try again after a few minutes",
            "",
            "5. Environment variable not working:",
            "   • Restart your terminal/IDE after setting",
            "   • Use 'echo $HF_TOKEN' to verify (Unix/Linux)",
            "   • Use 'echo %HF_TOKEN%' to verify (Windows)",
            "",
            "6. Still having issues?",
            "   • Try the 'Test Token' feature in this interface",
            "   • Check the Hugging Face documentation",
            "   • Verify all dependencies are installed correctly"
        ]

def get_hf_auth() -> HuggingFaceAuth:
    """
    Get HuggingFaceAuth instance (convenience function)
    
    Returns:
        HuggingFaceAuth instance
    """
    return HuggingFaceAuth()

def quick_token_test(token: str = None) -> Dict[str, Any]:
    """
    Quick token validation and connection test
    
    Args:
        token: Token to test (if None, tries to find from environment)
        
    Returns:
        Test results
    """
    auth = HuggingFaceAuth()
    
    if not token:
        sources = auth.get_token_sources()
        token = sources.get('active_token')
        if not token:
            return {
                'success': False,
                'error': 'No token found in environment or config files',
                'sources': sources
            }
    
    # Validate format
    validation = auth.validate_token_format(token)
    if not validation['valid']:
        return {
            'success': False,
            'error': f"Token format invalid: {', '.join(validation['errors'])}",
            'validation': validation
        }
    
    # Test connection
    connection = auth.test_token_connection(token)
    
    return {
        'success': connection['success'],
        'error': connection.get('error'),
        'user_info': connection.get('user_info'),
        'validation': validation,
        'connection': connection
    }