#!/usr/bin/env python3
"""
Railway Deployment Script for BIST DP-LSTM Trading System
Handles Railway-specific deployment, environment setup, and health checks
"""

import os
import sys
import subprocess
import logging
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class RailwayDeployment:
    """Railway-specific deployment manager"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.logger = self._setup_logging()
        
        # Railway configuration
        self.railway_project = "bist-dp-lstm-trading"
        self.service_name = "bist-trading-api"
        
        # Environment variables for Railway
        self.env_vars = {
            "ENVIRONMENT": "production",
            "LOG_LEVEL": "INFO",
            "PYTHONPATH": "/app",
            "PORT": "8000",
            
            # Database URLs (Railway will provide these)
            # "DATABASE_URL": "postgresql://...",  # Railway PostgreSQL
            # "REDIS_URL": "redis://...",           # Railway Redis
            
            # Application settings
            "MAX_WORKERS": "1",
            "WEB_CONCURRENCY": "1",
            "TIMEOUT": "300",
            
            # Trading system configuration
            "INITIAL_CAPITAL": "100000.0",
            "MAX_POSITIONS": "10",
            "COMMISSION_RATE": "0.001",
            "MAX_DAILY_LOSS": "0.05",
            
            # Signal generation settings
            "BUY_THRESHOLD": "0.65",
            "SELL_THRESHOLD": "0.65",
            "MIN_EXPECTED_RETURN": "0.012"
        }
        
        self.logger.info("Railway deployment manager initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Railway deployment"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("railway-deploy")
    
    def check_railway_cli(self) -> bool:
        """Check if Railway CLI is installed"""
        try:
            result = subprocess.run(
                ["railway", "--version"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ Railway CLI found: {result.stdout.strip()}")
                return True
            else:
                self.logger.error("❌ Railway CLI not found")
                return False
                
        except FileNotFoundError:
            self.logger.error("❌ Railway CLI not installed")
            return False
    
    def install_railway_cli(self) -> bool:
        """Install Railway CLI"""
        self.logger.info("📥 Installing Railway CLI...")
        
        try:
            if sys.platform == "darwin":  # macOS
                result = subprocess.run([
                    "curl", "-fsSL", "https://railway.app/install.sh"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Execute the install script
                    subprocess.run(["bash", "-c", result.stdout], check=True)
                    self.logger.info("✅ Railway CLI installed successfully")
                    return True
            else:
                self.logger.info("Please install Railway CLI manually:")
                self.logger.info("curl -fsSL https://railway.app/install.sh | sh")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to install Railway CLI: {e}")
            return False
    
    def railway_login(self) -> bool:
        """Login to Railway"""
        self.logger.info("🔐 Logging into Railway...")
        
        try:
            # Check if already logged in
            result = subprocess.run([
                "railway", "whoami"
            ], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                self.logger.info(f"✅ Already logged in: {result.stdout.strip()}")
                return True
            
            # Need to login
            self.logger.info("Please login to Railway...")
            result = subprocess.run([
                "railway", "login"
            ], check=True)
            
            self.logger.info("✅ Railway login successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Railway login failed: {e}")
            return False
    
    def create_or_connect_project(self) -> bool:
        """Create or connect to Railway project"""
        self.logger.info(f"🚂 Setting up Railway project: {self.railway_project}")
        
        try:
            # Try to create new project
            result = subprocess.run([
                "railway", "init", self.railway_project
            ], capture_output=True, text=True, check=False, cwd=self.project_root)
            
            if result.returncode == 0:
                self.logger.info("✅ New Railway project created")
                return True
            elif "already exists" in result.stderr.lower():
                # Project already exists, link to it
                result = subprocess.run([
                    "railway", "link"
                ], capture_output=True, text=True, input=f"{self.railway_project}\n", 
                check=False, cwd=self.project_root)
                
                if result.returncode == 0:
                    self.logger.info("✅ Connected to existing Railway project")
                    return True
            
            self.logger.error(f"❌ Failed to setup Railway project: {result.stderr}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Railway project setup failed: {e}")
            return False
    
    def setup_environment_variables(self) -> bool:
        """Set up environment variables in Railway"""
        self.logger.info("⚙️ Setting up environment variables...")
        
        try:
            for key, value in self.env_vars.items():
                result = subprocess.run([
                    "railway", "variables", "set", f"{key}={value}"
                ], capture_output=True, text=True, check=False, cwd=self.project_root)
                
                if result.returncode == 0:
                    self.logger.info(f"✅ Set {key}")
                else:
                    self.logger.warning(f"⚠️ Failed to set {key}: {result.stderr}")
            
            self.logger.info("✅ Environment variables configured")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Environment setup failed: {e}")
            return False
    
    def add_railway_services(self) -> bool:
        """Add required services (PostgreSQL, Redis)"""
        self.logger.info("🗄️ Setting up Railway services...")
        
        services = ["postgresql", "redis"]
        
        try:
            for service in services:
                self.logger.info(f"Adding {service} service...")
                
                result = subprocess.run([
                    "railway", "add", service
                ], capture_output=True, text=True, check=False, cwd=self.project_root)
                
                if result.returncode == 0:
                    self.logger.info(f"✅ Added {service} service")
                else:
                    self.logger.warning(f"⚠️ {service} service may already exist")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Service setup failed: {e}")
            return False
    
    def deploy_application(self) -> bool:
        """Deploy the application to Railway"""
        self.logger.info("🚀 Deploying to Railway...")
        
        try:
            # Deploy using Railway
            result = subprocess.run([
                "railway", "up", "--detach"
            ], capture_output=True, text=True, check=False, cwd=self.project_root)
            
            if result.returncode == 0:
                self.logger.info("✅ Deployment started successfully")
                self.logger.info(result.stdout)
                return True
            else:
                self.logger.error(f"❌ Deployment failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Deployment failed: {e}")
            return False
    
    def get_deployment_url(self) -> Optional[str]:
        """Get the deployed application URL"""
        try:
            result = subprocess.run([
                "railway", "status"
            ], capture_output=True, text=True, check=False, cwd=self.project_root)
            
            if result.returncode == 0:
                # Parse output to find URL
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'https://' in line and 'railway.app' in line:
                        url = line.split()[-1]
                        return url.strip()
            
        except Exception as e:
            self.logger.error(f"Error getting deployment URL: {e}")
        
        return None
    
    def check_deployment_health(self, url: str, max_attempts: int = 20) -> bool:
        """Check if deployment is healthy"""
        self.logger.info(f"🏥 Checking deployment health: {url}")
        
        for attempt in range(max_attempts):
            try:
                health_url = f"{url}/health"
                response = requests.get(health_url, timeout=10)
                
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get('status') in ['healthy', 'degraded']:
                        self.logger.info(f"✅ Deployment is healthy: {health_data.get('status')}")
                        return True
                        
                self.logger.info(f"⏳ Health check {attempt + 1}/{max_attempts}: {response.status_code}")
                
            except Exception as e:
                self.logger.info(f"⏳ Health check {attempt + 1}/{max_attempts}: {e}")
            
            time.sleep(15)  # Wait 15 seconds between attempts
        
        self.logger.error("❌ Health check failed after all attempts")
        return False
    
    def run_deployment(self) -> bool:
        """Run complete Railway deployment process"""
        self.logger.info("🚀 STARTING RAILWAY DEPLOYMENT")
        self.logger.info("=" * 60)
        
        steps = [
            ("Check Railway CLI", self.check_railway_cli),
            ("Railway Login", self.railway_login),
            ("Setup Project", self.create_or_connect_project),
            ("Add Services", self.add_railway_services),
            ("Setup Environment", self.setup_environment_variables),
            ("Deploy Application", self.deploy_application)
        ]
        
        for step_name, step_func in steps:
            self.logger.info(f"\n📋 Step: {step_name}")
            if not step_func():
                if step_name == "Check Railway CLI":
                    # Try to install CLI
                    if not self.install_railway_cli():
                        self.logger.error(f"❌ Deployment failed at step: {step_name}")
                        return False
                else:
                    self.logger.error(f"❌ Deployment failed at step: {step_name}")
                    return False
        
        # Wait for deployment and get URL
        self.logger.info("\n⏳ Waiting for deployment to be ready...")
        time.sleep(30)
        
        deployment_url = self.get_deployment_url()
        
        if deployment_url:
            self.logger.info(f"🌐 Deployment URL: {deployment_url}")
            
            # Check health
            if self.check_deployment_health(deployment_url):
                self._display_success_summary(deployment_url)
                return True
        else:
            self.logger.warning("⚠️ Could not retrieve deployment URL automatically")
            self.logger.info("Check Railway dashboard: https://railway.app/dashboard")
        
        return True
    
    def _display_success_summary(self, url: str):
        """Display deployment success summary"""
        print("\n" + "="*80)
        print("🎉 RAILWAY DEPLOYMENT SUCCESSFUL!")
        print("="*80)
        
        print(f"🌐 Application URL: {url}")
        print(f"📚 API Documentation: {url}/docs")
        print(f"❤️ Health Check: {url}/health")
        print(f"📊 System Metrics: {url}/metrics/system")
        print(f"💼 Portfolio Summary: {url}/portfolio/summary")
        
        print(f"\n🔧 MANAGEMENT:")
        print(f"   Railway Dashboard: https://railway.app/dashboard")
        print(f"   View Logs: railway logs")
        print(f"   Check Status: railway status")
        print(f"   Redeploy: railway up")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Test all API endpoints")
        print(f"   2. Configure real BIST data sources") 
        print(f"   3. Monitor system performance")
        print(f"   4. Set up alerts and notifications")
        
        print("="*80)


def main():
    """Main deployment script"""
    railway_deploy = RailwayDeployment()
    
    try:
        success = railway_deploy.run_deployment()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Deployment failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
