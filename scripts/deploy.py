#!/usr/bin/env python3
"""
BIST DP-LSTM Trading System - Production Deployment Script
Handles deployment, health checks, and environment setup
"""

import os
import sys
import subprocess
import logging
import argparse
import time
import json
import shutil
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class DeploymentManager:
    """Handles system deployment and configuration"""
    
    def __init__(self, target: str = "local"):
        self.target = target
        self.project_root = Path(__file__).parent.parent
        self.logger = self._setup_logging()
        self.deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Configuration
        self.config = {
            'api_port': 8000,
            'health_check_timeout': 120,  # seconds
            'health_check_interval': 5,   # seconds
            'database_port': 5432,
            'redis_port': 6379,
            'influxdb_port': 8086,
            'grafana_port': 3000
        }
        
        self.logger.info(f"Deployment Manager initialized for target: {target}")
        self.logger.info(f"Deployment ID: {self.deployment_id}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for deployment"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / f"deployment_{datetime.now().strftime('%Y%m%d')}.log")
            ]
        )
        return logging.getLogger("deployment")
    
    def run_command(self, command: str, check: bool = True, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Execute shell command with error handling"""
        self.logger.info(f"Executing: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd or self.project_root,
                check=check
            )
            
            if result.stdout:
                self.logger.debug(f"STDOUT: {result.stdout}")
            if result.stderr and result.returncode == 0:
                self.logger.debug(f"STDERR: {result.stderr}")
            
            return result
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {command}")
            self.logger.error(f"Return code: {e.returncode}")
            self.logger.error(f"STDOUT: {e.stdout}")
            self.logger.error(f"STDERR: {e.stderr}")
            if check:
                raise
            return e
    
    def check_prerequisites(self) -> bool:
        """Check system prerequisites for deployment"""
        self.logger.info("🔍 Checking deployment prerequisites...")
        
        prerequisites = {
            'docker': 'docker --version',
            'docker-compose': 'docker-compose --version',
            'curl': 'curl --version'
        }
        
        missing = []
        for name, command in prerequisites.items():
            result = self.run_command(command, check=False)
            if result.returncode != 0:
                missing.append(name)
                self.logger.error(f"❌ Missing: {name}")
            else:
                self.logger.info(f"✅ Found: {name}")
        
        if missing:
            self.logger.error(f"Missing prerequisites: {', '.join(missing)}")
            return False
        
        # Check Docker daemon
        result = self.run_command("docker info", check=False)
        if result.returncode != 0:
            self.logger.error("❌ Docker daemon not running")
            return False
        
        self.logger.info("✅ All prerequisites satisfied")
        return True
    
    def prepare_environment(self) -> bool:
        """Prepare deployment environment"""
        self.logger.info("📝 Preparing deployment environment...")
        
        try:
            # Create necessary directories
            directories = ['logs', 'data', 'models', 'cache', 'config']
            for dir_name in directories:
                dir_path = self.project_root / dir_name
                dir_path.mkdir(exist_ok=True)
                self.logger.info(f"📁 Created directory: {dir_name}")
            
            # Create basic config files if they don't exist
            self._create_config_files()
            
            # Set permissions
            if os.name != 'nt':  # Not Windows
                self.run_command("chmod +x scripts/*.py", check=False)
            
            self.logger.info("✅ Environment prepared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare environment: {e}")
            return False
    
    def _create_config_files(self):
        """Create necessary configuration files"""
        config_dir = self.project_root / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Redis configuration
        redis_config = """# Redis configuration for BIST Trading System
bind 0.0.0.0
port 6379
timeout 300
tcp-keepalive 60
maxmemory 256mb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec
"""
        (config_dir / "redis.conf").write_text(redis_config)
        
        # Nginx configuration (basic)
        nginx_dir = config_dir / "nginx"
        nginx_dir.mkdir(exist_ok=True)
        
        nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server bist-trading-api:8000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
        
        location /health {
            proxy_pass http://api_backend/health;
            access_log off;
        }
    }
}
"""
        (nginx_dir / "nginx.conf").write_text(nginx_config)
        
        self.logger.info("📝 Configuration files created")
    
    def build_application(self) -> bool:
        """Build Docker images"""
        self.logger.info("🔨 Building application...")
        
        try:
            # Build main application
            self.run_command("docker-compose build --no-cache")
            self.logger.info("✅ Application built successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to build application: {e}")
            return False
    
    def deploy_services(self) -> bool:
        """Deploy all services using docker-compose"""
        self.logger.info("🚀 Deploying services...")
        
        try:
            # Stop existing services
            self.logger.info("Stopping existing services...")
            self.run_command("docker-compose down", check=False)
            
            # Start services
            self.logger.info("Starting services...")
            self.run_command("docker-compose up -d")
            
            # Wait for services to initialize
            self.logger.info("Waiting for services to initialize...")
            time.sleep(30)
            
            self.logger.info("✅ Services deployed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to deploy services: {e}")
            return False
    
    def check_service_health(self) -> Dict[str, bool]:
        """Check health of all deployed services"""
        self.logger.info("🏥 Checking service health...")
        
        services = {
            'api': f"http://localhost:{self.config['api_port']}/health",
            'postgres': f"localhost:{self.config['database_port']}",
            'redis': f"localhost:{self.config['redis_port']}",
            'influxdb': f"http://localhost:{self.config['influxdb_port']}/health"
        }
        
        health_status = {}
        
        for service_name, endpoint in services.items():
            if service_name == 'api' or service_name == 'influxdb':
                # HTTP health check
                health_status[service_name] = self._check_http_health(service_name, endpoint)
            elif service_name == 'postgres':
                # Database health check
                health_status[service_name] = self._check_postgres_health()
            elif service_name == 'redis':
                # Redis health check
                health_status[service_name] = self._check_redis_health()
            else:
                health_status[service_name] = False
        
        # Summary
        healthy_services = sum(health_status.values())
        total_services = len(health_status)
        
        self.logger.info(f"Health check results: {healthy_services}/{total_services} services healthy")
        for service, status in health_status.items():
            status_icon = "✅" if status else "❌"
            self.logger.info(f"  {status_icon} {service}")
        
        return health_status
    
    def _check_http_health(self, service_name: str, url: str) -> bool:
        """Check HTTP service health"""
        max_attempts = self.config['health_check_timeout'] // self.config['health_check_interval']
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    self.logger.info(f"✅ {service_name} is healthy")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            if attempt < max_attempts - 1:
                time.sleep(self.config['health_check_interval'])
        
        self.logger.warning(f"❌ {service_name} health check failed")
        return False
    
    def _check_postgres_health(self) -> bool:
        """Check PostgreSQL health"""
        try:
            result = self.run_command(
                "docker exec bist_trading_postgres pg_isready -U postgres -d bist_trading",
                check=False
            )
            return result.returncode == 0
        except:
            return False
    
    def _check_redis_health(self) -> bool:
        """Check Redis health"""
        try:
            result = self.run_command(
                "docker exec bist_trading_redis redis-cli ping",
                check=False
            )
            return result.returncode == 0 and "PONG" in result.stdout
        except:
            return False
    
    def run_integration_tests(self) -> bool:
        """Run basic integration tests"""
        self.logger.info("🧪 Running integration tests...")
        
        tests = [
            self._test_api_endpoints,
            self._test_signal_generation,
            self._test_metrics_collection
        ]
        
        passed = 0
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                self.logger.error(f"Test failed: {e}")
        
        self.logger.info(f"Integration tests: {passed}/{len(tests)} passed")
        return passed == len(tests)
    
    def _test_api_endpoints(self) -> bool:
        """Test API endpoints"""
        endpoints = [
            "/",
            "/health",
            "/metrics/system",
            "/portfolio/summary"
        ]
        
        base_url = f"http://localhost:{self.config['api_port']}"
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
                if response.status_code != 200:
                    self.logger.error(f"API endpoint {endpoint} failed: {response.status_code}")
                    return False
            except Exception as e:
                self.logger.error(f"API endpoint {endpoint} error: {e}")
                return False
        
        self.logger.info("✅ API endpoints test passed")
        return True
    
    def _test_signal_generation(self) -> bool:
        """Test signal generation"""
        try:
            url = f"http://localhost:{self.config['api_port']}/signals/generate"
            data = {"symbol": "AKBNK", "include_features": True}
            
            response = requests.post(url, json=data, timeout=15)
            if response.status_code != 200:
                self.logger.error(f"Signal generation failed: {response.status_code}")
                return False
            
            result = response.json()
            if not result.get('symbol') or not result.get('action'):
                self.logger.error("Invalid signal response format")
                return False
            
            self.logger.info("✅ Signal generation test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Signal generation test error: {e}")
            return False
    
    def _test_metrics_collection(self) -> bool:
        """Test metrics collection"""
        try:
            url = f"http://localhost:{self.config['api_port']}/metrics/system"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                self.logger.error(f"Metrics collection failed: {response.status_code}")
                return False
            
            metrics = response.json()
            required_metrics = ['cpu_usage_pct', 'memory_usage_pct', 'system_uptime']
            
            for metric in required_metrics:
                if metric not in metrics:
                    self.logger.error(f"Missing metric: {metric}")
                    return False
            
            self.logger.info("✅ Metrics collection test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Metrics test error: {e}")
            return False
    
    def display_deployment_summary(self, health_status: Dict[str, bool], tests_passed: bool):
        """Display deployment summary"""
        print("\n" + "="*80)
        print("🎉 DEPLOYMENT SUMMARY")
        print("="*80)
        
        print(f"Deployment ID: {self.deployment_id}")
        print(f"Target Environment: {self.target}")
        print(f"Deployment Time: {datetime.now()}")
        
        print(f"\n📊 SERVICE STATUS:")
        for service, status in health_status.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {service.title()}")
        
        healthy_count = sum(health_status.values())
        print(f"\nServices: {healthy_count}/{len(health_status)} healthy")
        print(f"Integration Tests: {'✅ PASSED' if tests_passed else '❌ FAILED'}")
        
        if healthy_count == len(health_status) and tests_passed:
            print(f"\n🎉 DEPLOYMENT SUCCESSFUL!")
            print(f"\n📡 ACCESS POINTS:")
            print(f"   🌐 API Documentation: http://localhost:{self.config['api_port']}/docs")
            print(f"   ❤️  Health Check: http://localhost:{self.config['api_port']}/health")
            print(f"   📊 Grafana Dashboard: http://localhost:{self.config['grafana_port']}")
            print(f"   🔍 System Metrics: http://localhost:{self.config['api_port']}/metrics/system")
        else:
            print(f"\n⚠️  DEPLOYMENT COMPLETED WITH ISSUES!")
            print(f"   Check logs for details: logs/deployment_*.log")
        
        print("="*80)
    
    def deploy(self) -> bool:
        """Main deployment orchestration"""
        self.logger.info(f"🚀 Starting deployment to {self.target}")
        
        steps = [
            ("Prerequisites", self.check_prerequisites),
            ("Environment", self.prepare_environment),
            ("Build", self.build_application),
            ("Deploy Services", self.deploy_services)
        ]
        
        for step_name, step_func in steps:
            self.logger.info(f"📋 Step: {step_name}")
            if not step_func():
                self.logger.error(f"❌ Deployment failed at step: {step_name}")
                return False
        
        # Health checks and tests
        health_status = self.check_service_health()
        tests_passed = self.run_integration_tests()
        
        # Display summary
        self.display_deployment_summary(health_status, tests_passed)
        
        # Final result
        all_healthy = all(health_status.values())
        deployment_success = all_healthy and tests_passed
        
        if deployment_success:
            self.logger.info("🎉 Deployment completed successfully!")
        else:
            self.logger.error("❌ Deployment completed with issues")
        
        return deployment_success


def main():
    """Main deployment script entry point"""
    parser = argparse.ArgumentParser(
        description='BIST DP-LSTM Trading System Deployment Script'
    )
    parser.add_argument(
        '--target', 
        choices=['local', 'staging', 'production'], 
        default='local',
        help='Deployment target environment'
    )
    parser.add_argument(
        '--skip-tests', 
        action='store_true',
        help='Skip integration tests'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create deployment manager and run deployment
    deployment_manager = DeploymentManager(target=args.target)
    
    try:
        success = deployment_manager.deploy()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Deployment failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
