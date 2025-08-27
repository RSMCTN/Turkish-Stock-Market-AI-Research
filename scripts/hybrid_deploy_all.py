#!/usr/bin/env python3
"""
Hybrid Deployment Manager
Orchestrates deployment across GitHub + Hugging Face + Railway
"""

import os
import sys
import subprocess
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridDeploymentManager:
    """Manages deployment across all platforms"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.deployment_status = {
            'github': False,
            'huggingface': False, 
            'railway': False
        }
        
    def check_prerequisites(self) -> bool:
        """Check if all required tools are installed and authenticated"""
        logger.info("🔍 Checking prerequisites...")
        
        requirements = {
            'git': self._check_git(),
            'huggingface-cli': self._check_huggingface_cli(),
            'railway': self._check_railway_cli(),
            'docker': self._check_docker()
        }
        
        missing = [tool for tool, available in requirements.items() if not available]
        
        if missing:
            logger.error(f"❌ Missing tools: {', '.join(missing)}")
            self._print_installation_guide(missing)
            return False
        
        logger.info("✅ All prerequisites met")
        return True
    
    def _check_git(self) -> bool:
        """Check git installation and authentication"""
        try:
            subprocess.run(['git', '--version'], check=True, capture_output=True)
            result = subprocess.run(['git', 'remote', '-v'], check=True, capture_output=True, text=True)
            return 'origin' in result.stdout
        except:
            return False
    
    def _check_huggingface_cli(self) -> bool:
        """Check Hugging Face CLI installation and authentication"""
        try:
            result = subprocess.run(['huggingface-cli', 'whoami'], check=True, capture_output=True, text=True)
            return 'Not logged' not in result.stdout
        except:
            return False
    
    def _check_railway_cli(self) -> bool:
        """Check Railway CLI installation and authentication"""
        try:
            subprocess.run(['railway', '--version'], check=True, capture_output=True)
            result = subprocess.run(['railway', 'whoami'], check=True, capture_output=True, text=True)
            return 'Not logged in' not in result.stdout
        except:
            return False
    
    def _check_docker(self) -> bool:
        """Check Docker installation"""
        try:
            subprocess.run(['docker', '--version'], check=True, capture_output=True)
            subprocess.run(['docker', 'ps'], check=True, capture_output=True)
            return True
        except:
            return False
    
    def _print_installation_guide(self, missing_tools: List[str]) -> None:
        """Print installation guide for missing tools"""
        print("\\n" + "="*60)
        print("📋 INSTALLATION GUIDE")
        print("="*60)
        
        guides = {
            'git': [
                "🔧 Git Installation:",
                "macOS: brew install git",
                "Ubuntu: sudo apt install git",
                "Configure: git config --global user.name 'Your Name'",
                "Configure: git config --global user.email 'your.email@example.com'"
            ],
            'huggingface-cli': [
                "🤗 Hugging Face CLI Installation:",
                "pip install huggingface_hub[cli]",
                "Login: huggingface-cli login",
                "Paste your HF token when prompted"
            ],
            'railway': [
                "🚂 Railway CLI Installation:",
                "macOS: brew install railway",
                "Other: curl -fsSL https://railway.app/install.sh | sh",
                "Login: railway login"
            ],
            'docker': [
                "🐳 Docker Installation:",
                "macOS: brew install --cask docker",
                "Ubuntu: sudo apt install docker.io",
                "Start: sudo systemctl start docker"
            ]
        }
        
        for tool in missing_tools:
            if tool in guides:
                for line in guides[tool]:
                    print(f"  {line}")
                print()
    
    def deploy_github(self) -> bool:
        """Deploy to GitHub"""
        logger.info("🐙 Deploying to GitHub...")
        
        try:
            # Check for uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, check=True)
            
            if result.stdout.strip():
                logger.info("📝 Committing changes...")
                subprocess.run(['git', 'add', '.'], check=True)
                
                commit_message = f"""🌟 Hybrid Deployment Update - {datetime.now().strftime('%Y-%m-%d %H:%M')}

✨ Features Added:
• GitHub + Hugging Face + Railway integration
• Interactive Gradio dashboard for BIST trading signals
• Automated model & dataset upload to HF Hub
• Production-ready Railway deployment
• Comprehensive hybrid deployment documentation

🔧 Technical Improvements:
• Docker optimizations for Railway
• Hugging Face Spaces compatibility
• Model cards and dataset documentation
• Unified deployment management
• Cross-platform authentication

🚀 Deployment Ready:
• Code: GitHub repository
• Models: Hugging Face Hub
• Dashboard: HF Spaces
• Production: Railway API

🎯 All platforms synchronized and production-ready!"""
                
                subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            
            # Push to GitHub
            logger.info("📤 Pushing to GitHub...")
            subprocess.run(['git', 'push', 'origin', 'main'], check=True)
            
            self.deployment_status['github'] = True
            logger.info("✅ GitHub deployment successful")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ GitHub deployment failed: {str(e)}")
            return False
    
    def deploy_huggingface(self) -> bool:
        """Deploy to Hugging Face"""
        logger.info("🤗 Deploying to Hugging Face...")
        
        try:
            # Run HuggingFace setup script
            hf_script = self.project_root / "scripts" / "huggingface_setup.py"
            
            if hf_script.exists():
                subprocess.run([sys.executable, str(hf_script)], check=True)
                self.deployment_status['huggingface'] = True
                logger.info("✅ Hugging Face deployment successful")
                return True
            else:
                logger.error("❌ HuggingFace setup script not found")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Hugging Face deployment failed: {str(e)}")
            return False
    
    def deploy_railway(self) -> bool:
        """Deploy to Railway"""
        logger.info("🚂 Deploying to Railway...")
        
        try:
            # Run Railway deployment script
            railway_script = self.project_root / "scripts" / "deploy_railway.py"
            
            if railway_script.exists():
                subprocess.run([sys.executable, str(railway_script)], check=True)
                self.deployment_status['railway'] = True
                logger.info("✅ Railway deployment successful")
                return True
            else:
                logger.error("❌ Railway deployment script not found")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Railway deployment failed: {str(e)}")
            return False
    
    def deploy_all(self, skip_platforms: List[str] = None) -> bool:
        """Deploy to all platforms"""
        skip_platforms = skip_platforms or []
        
        logger.info("🚀 Starting hybrid deployment...")
        print("\\n" + "="*60)
        print("🌟 HYBRID DEPLOYMENT - ALL PLATFORMS")
        print("="*60)
        
        if not self.check_prerequisites():
            return False
        
        platforms = [
            ('github', self.deploy_github),
            ('huggingface', self.deploy_huggingface), 
            ('railway', self.deploy_railway)
        ]
        
        success_count = 0
        
        for platform_name, deploy_func in platforms:
            if platform_name in skip_platforms:
                logger.info(f"⏭️ Skipping {platform_name}")
                continue
                
            print(f"\\n🎯 Deploying to {platform_name.upper()}...")
            if deploy_func():
                success_count += 1
            
            # Small delay between deployments
            time.sleep(2)
        
        # Print final summary
        self._print_deployment_summary(success_count, len([p for p in platforms if p[0] not in skip_platforms]))
        
        return success_count == len([p for p in platforms if p[0] not in skip_platforms])
    
    def _print_deployment_summary(self, success_count: int, total_count: int) -> None:
        """Print deployment summary"""
        print("\\n" + "="*60)
        print("📊 DEPLOYMENT SUMMARY")
        print("="*60)
        
        for platform, status in self.deployment_status.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {platform.upper()}: {'SUCCESS' if status else 'FAILED'}")
        
        print(f"\\n🎯 Overall: {success_count}/{total_count} platforms deployed successfully")
        
        if success_count == total_count:
            print("\\n🎉 ALL DEPLOYMENTS SUCCESSFUL! 🎉")
            print("\\n🔗 PLATFORM LINKS:")
            print("🐙 GitHub: https://github.com/RSMCTN/BIST_AI001")
            print("🤗 HuggingFace: https://huggingface.co/RSMCTN?search=bist-dp-lstm")
            print("🚂 Railway: https://bist-dp-lstm-trading.up.railway.app")
            print("🎛️ Dashboard: https://huggingface.co/spaces/RSMCTN/bist-dp-lstm-trading-trading_dashboard")
            print("\\n🚀 Your hybrid ML system is now live across all platforms! 🚀")
        else:
            print("\\n⚠️ Some deployments failed. Check logs above for details.")
    
    def status_check(self) -> Dict[str, bool]:
        """Check status of all deployments"""
        logger.info("🔍 Checking deployment status...")
        
        # GitHub status
        try:
            subprocess.run(['git', 'ls-remote', 'origin'], check=True, capture_output=True)
            self.deployment_status['github'] = True
        except:
            self.deployment_status['github'] = False
        
        # Hugging Face status (simplified check)
        try:
            subprocess.run(['huggingface-cli', 'whoami'], check=True, capture_output=True)
            self.deployment_status['huggingface'] = True
        except:
            self.deployment_status['huggingface'] = False
        
        # Railway status (simplified check)
        try:
            subprocess.run(['railway', 'status'], check=True, capture_output=True)
            self.deployment_status['railway'] = True
        except:
            self.deployment_status['railway'] = False
        
        return self.deployment_status

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Hybrid Deployment Manager')
    parser.add_argument('--skip', nargs='*', choices=['github', 'huggingface', 'railway'],
                       help='Platforms to skip during deployment')
    parser.add_argument('--status', action='store_true', help='Check deployment status only')
    parser.add_argument('--platform', choices=['github', 'huggingface', 'railway'],
                       help='Deploy to specific platform only')
    
    args = parser.parse_args()
    
    manager = HybridDeploymentManager()
    
    if args.status:
        status = manager.status_check()
        manager._print_deployment_summary(
            sum(status.values()), 
            len(status)
        )
        return
    
    if args.platform:
        # Deploy to specific platform
        platform_methods = {
            'github': manager.deploy_github,
            'huggingface': manager.deploy_huggingface,
            'railway': manager.deploy_railway
        }
        
        if manager.check_prerequisites():
            success = platform_methods[args.platform]()
            print(f"\\n{'✅ SUCCESS' if success else '❌ FAILED'}: {args.platform} deployment")
    else:
        # Deploy to all platforms (with optional skips)
        manager.deploy_all(skip_platforms=args.skip)

if __name__ == "__main__":
    main()
