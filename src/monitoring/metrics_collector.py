"""
Advanced Metrics Collection & Monitoring System for BIST DP-LSTM Trading System
Collects system metrics, trading metrics, and provides alerting capabilities
"""

import asyncio
import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from enum import Enum
import json


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SystemMetric:
    """Individual system metric"""
    timestamp: datetime
    metric_name: str
    metric_value: float
    metric_unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_unit': self.metric_unit,
            'tags': self.tags
        }


@dataclass
class Alert:
    """System alert"""
    timestamp: datetime
    level: AlertLevel
    metric_name: str
    current_value: float
    threshold: float
    message: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'message': self.message,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


class MetricsCollector:
    """
    Advanced metrics collection and monitoring system
    
    Features:
    - System resource monitoring (CPU, memory, disk, network)
    - Trading-specific metrics (signals, executions, P&L)
    - Real-time alerting with configurable thresholds
    - Historical metric storage with automatic cleanup
    - Performance analytics and trend analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize metrics collector"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # System state
        self.start_time = datetime.now()
        self.is_running = False
        self._collection_task: Optional[asyncio.Task] = None
        
        # Metric storage (thread-safe)
        self._metrics_lock = threading.RLock()
        self.metrics_history: deque = deque(maxlen=10000)  # Last 10k metrics
        self.current_metrics: Dict[str, SystemMetric] = {}
        self.metric_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Alert system
        self.alert_thresholds = {
            'system.cpu.usage': {'warning': 70.0, 'critical': 85.0},
            'system.memory.usage': {'warning': 75.0, 'critical': 90.0},
            'api.response_time': {'warning': 500.0, 'critical': 1000.0},  # milliseconds
            'api.error_rate': {'warning': 2.0, 'critical': 5.0},  # percentage
            'trading.prediction_latency': {'warning': 300.0, 'critical': 500.0},  # milliseconds
            'trading.portfolio_drawdown': {'warning': 10.0, 'critical': 15.0}  # percentage
        }
        
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []
        
        # Collection settings
        self.collection_interval = self.config.get('collection_interval', 30)  # seconds
        self.alert_cooldown = self.config.get('alert_cooldown', 300)  # seconds
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Trading metrics integration
        self.trading_signal_generator = None
        self.trading_paper_engine = None
        
        self.logger.info("MetricsCollector initialized")
    
    def set_trading_components(self, signal_generator=None, paper_engine=None):
        """Set trading components for metrics collection"""
        self.trading_signal_generator = signal_generator
        self.trading_paper_engine = paper_engine
        self.logger.info("Trading components set for metrics collection")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)
    
    async def start_collection(self):
        """Start background metrics collection"""
        if self.is_running:
            self.logger.warning("Metrics collection already running")
            return
        
        self.is_running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        self.logger.info(f"Started metrics collection (interval: {self.collection_interval}s)")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped metrics collection")
    
    async def _collection_loop(self):
        """Main metrics collection loop"""
        while self.is_running:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}", exc_info=True)
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_all_metrics(self):
        """Collect all types of metrics"""
        timestamp = datetime.now()
        
        # Collect system metrics
        await self._collect_system_metrics(timestamp)
        
        # Collect trading metrics
        await self._collect_trading_metrics(timestamp)
        
        # Collect API metrics (if available)
        await self._collect_api_metrics(timestamp)
        
        # Check alert conditions
        await self._check_alert_thresholds(timestamp)
        
        # Cleanup old data
        await self._cleanup_old_data(timestamp)
    
    async def _collect_system_metrics(self, timestamp: datetime):
        """Collect system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self._add_metric("system.cpu.usage", cpu_percent, "%", timestamp)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self._add_metric("system.memory.usage", memory.percent, "%", timestamp)
            self._add_metric("system.memory.available", memory.available / (1024**3), "GB", timestamp)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_pct = (disk.used / disk.total) * 100
            self._add_metric("system.disk.usage", disk_usage_pct, "%", timestamp)
            self._add_metric("system.disk.free", disk.free / (1024**3), "GB", timestamp)
            
            # Network metrics (if available)
            try:
                net_io = psutil.net_io_counters()
                self._add_metric("system.network.bytes_sent", net_io.bytes_sent, "bytes", timestamp)
                self._add_metric("system.network.bytes_recv", net_io.bytes_recv, "bytes", timestamp)
            except:
                pass  # Network metrics not available on all systems
            
            # Process-specific metrics
            process = psutil.Process()
            self._add_metric("process.memory_rss", process.memory_info().rss / (1024**2), "MB", timestamp)
            self._add_metric("process.cpu_percent", process.cpu_percent(), "%", timestamp)
            
            # System uptime
            uptime_seconds = (timestamp - self.start_time).total_seconds()
            self._add_metric("system.uptime", uptime_seconds, "seconds", timestamp)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_trading_metrics(self, timestamp: datetime):
        """Collect trading-specific metrics"""
        try:
            # Signal generator metrics
            if self.trading_signal_generator:
                try:
                    stats = self.trading_signal_generator.get_daily_stats()
                    self._add_metric("trading.signals.total_today", stats.get("total_signals_today", 0), "count", timestamp)
                    self._add_metric("trading.signals.unique_symbols", stats.get("unique_symbols", 0), "count", timestamp)
                    
                    # Recent prediction performance (if available)
                    if hasattr(self.trading_signal_generator, 'prediction_history') and self.trading_signal_generator.prediction_history:
                        recent_predictions = self.trading_signal_generator.prediction_history[-10:]
                        avg_confidence = sum(p['prediction']['confidence'] for p in recent_predictions) / len(recent_predictions)
                        self._add_metric("trading.prediction.avg_confidence", avg_confidence, "ratio", timestamp)
                
                except Exception as e:
                    self.logger.debug(f"Error collecting signal generator metrics: {e}")
            
            # Paper trading engine metrics
            if self.trading_paper_engine:
                try:
                    status = self.trading_paper_engine.get_current_status()
                    portfolio = status["portfolio_summary"]
                    
                    # Portfolio metrics
                    self._add_metric("trading.portfolio.value", portfolio["capital"]["current_value"], "TL", timestamp)
                    self._add_metric("trading.portfolio.return_pct", portfolio["capital"]["total_return_pct"], "%", timestamp)
                    self._add_metric("trading.portfolio.positions_count", portfolio["positions"]["count"], "count", timestamp)
                    self._add_metric("trading.portfolio.unrealized_pnl", portfolio["positions"]["unrealized_pnl"], "TL", timestamp)
                    
                    # Trading metrics
                    self._add_metric("trading.trades.total", portfolio["trades"]["total_count"], "count", timestamp)
                    self._add_metric("trading.trades.win_rate", portfolio["trades"]["win_rate"], "ratio", timestamp)
                    
                    # Risk metrics
                    self._add_metric("trading.risk.drawdown", portfolio["risk"]["max_drawdown"], "ratio", timestamp)
                    
                    # Execution metrics
                    exec_stats = status.get("recent_execution_stats", {})
                    for metric, values in exec_stats.items():
                        if values:
                            avg_value = sum(values) / len(values)
                            self._add_metric(f"trading.execution.avg_{metric}", avg_value, "ratio", timestamp)
                
                except Exception as e:
                    self.logger.debug(f"Error collecting paper trading metrics: {e}")
        
        except Exception as e:
            self.logger.error(f"Error collecting trading metrics: {e}")
    
    async def _collect_api_metrics(self, timestamp: datetime):
        """Collect API performance metrics"""
        try:
            # This would be integrated with the FastAPI application
            # For now, we'll collect basic metrics that might be available
            
            # Placeholder metrics - would be populated by API middleware
            self._add_metric("api.requests.total", 0, "count", timestamp)
            self._add_metric("api.response_time.avg", 0, "ms", timestamp)
            self._add_metric("api.error_rate", 0, "%", timestamp)
        
        except Exception as e:
            self.logger.error(f"Error collecting API metrics: {e}")
    
    def _add_metric(self, name: str, value: float, unit: str, timestamp: datetime, tags: Optional[Dict[str, str]] = None):
        """Add metric to collection (thread-safe)"""
        metric = SystemMetric(
            timestamp=timestamp,
            metric_name=name,
            metric_value=float(value),
            metric_unit=unit,
            tags=tags or {}
        )
        
        with self._metrics_lock:
            # Add to history
            self.metrics_history.append(metric)
            
            # Update current metrics
            self.current_metrics[name] = metric
            
            # Update trend data
            self.metric_trends[name].append((timestamp, float(value)))
    
    async def _check_alert_thresholds(self, timestamp: datetime):
        """Check metrics against alert thresholds"""
        with self._metrics_lock:
            for metric_name, metric in self.current_metrics.items():
                if metric_name in self.alert_thresholds:
                    thresholds = self.alert_thresholds[metric_name]
                    
                    # Check critical threshold
                    if metric.metric_value > thresholds.get('critical', float('inf')):
                        await self._trigger_alert(
                            metric_name, metric.metric_value, thresholds['critical'],
                            AlertLevel.CRITICAL, timestamp
                        )
                    # Check warning threshold
                    elif metric.metric_value > thresholds.get('warning', float('inf')):
                        await self._trigger_alert(
                            metric_name, metric.metric_value, thresholds['warning'],
                            AlertLevel.WARNING, timestamp
                        )
                    else:
                        # Check if we need to resolve an existing alert
                        await self._resolve_alert(metric_name, timestamp)
    
    async def _trigger_alert(self, metric_name: str, current_value: float, threshold: float, 
                           level: AlertLevel, timestamp: datetime):
        """Trigger an alert if conditions are met"""
        
        # Check cooldown period
        if metric_name in self.last_alert_times:
            if (timestamp - self.last_alert_times[metric_name]).total_seconds() < self.alert_cooldown:
                return
        
        # Check if alert already exists and is not resolved
        if metric_name in self.active_alerts and not self.active_alerts[metric_name].resolved:
            # Update existing alert if level escalated
            existing_alert = self.active_alerts[metric_name]
            if level.value == 'critical' and existing_alert.level.value == 'warning':
                existing_alert.level = level
                existing_alert.current_value = current_value
                existing_alert.threshold = threshold
                existing_alert.timestamp = timestamp
                await self._notify_alert_callbacks(existing_alert)
            return
        
        # Create new alert
        message = f"{metric_name} {level.value.upper()}: {current_value:.2f} exceeds threshold {threshold:.2f}"
        alert = Alert(
            timestamp=timestamp,
            level=level,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            message=message
        )
        
        # Store alert
        self.active_alerts[metric_name] = alert
        self.alert_history.append(alert)
        self.last_alert_times[metric_name] = timestamp
        
        # Notify
        self.logger.warning(f"ALERT: {message}")
        await self._notify_alert_callbacks(alert)
    
    async def _resolve_alert(self, metric_name: str, timestamp: datetime):
        """Resolve an active alert"""
        if metric_name in self.active_alerts and not self.active_alerts[metric_name].resolved:
            alert = self.active_alerts[metric_name]
            alert.resolved = True
            alert.resolved_at = timestamp
            
            self.logger.info(f"RESOLVED: Alert for {metric_name}")
            await self._notify_alert_callbacks(alert)
    
    async def _notify_alert_callbacks(self, alert: Alert):
        """Notify all registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    async def _cleanup_old_data(self, timestamp: datetime):
        """Clean up old metrics and alerts"""
        # This runs during each collection cycle, but we only actually clean up periodically
        if hasattr(self, '_last_cleanup') and (timestamp - self._last_cleanup).total_seconds() < 3600:
            return
        
        self._last_cleanup = timestamp
        
        with self._metrics_lock:
            # Clean up resolved alerts older than 24 hours
            cutoff_time = timestamp - timedelta(hours=24)
            self.alert_history = deque([
                alert for alert in self.alert_history
                if not (alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time)
            ], maxlen=1000)
        
        self.logger.debug("Completed metrics cleanup")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        with self._metrics_lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'metrics': {name: metric.to_dict() for name, metric in self.current_metrics.items()},
                'collection_active': self.is_running
            }
    
    def get_metric_history(self, metric_name: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._metrics_lock:
            return [
                metric.to_dict() for metric in self.metrics_history
                if metric.metric_name == metric_name and metric.timestamp >= cutoff_time
            ]
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [
            alert.to_dict() for alert in self.active_alerts.values()
            if not alert.resolved
        ]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert.to_dict() for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
    
    def get_metric_trends(self, metric_name: str) -> Dict[str, Any]:
        """Get trend analysis for a metric"""
        if metric_name not in self.metric_trends:
            return {}
        
        with self._metrics_lock:
            trend_data = list(self.metric_trends[metric_name])
            
            if len(trend_data) < 2:
                return {'trend': 'insufficient_data'}
            
            values = [value for _, value in trend_data]
            
            # Simple trend analysis
            recent_avg = sum(values[-10:]) / min(10, len(values))
            overall_avg = sum(values) / len(values)
            
            trend = 'stable'
            if recent_avg > overall_avg * 1.1:
                trend = 'increasing'
            elif recent_avg < overall_avg * 0.9:
                trend = 'decreasing'
            
            return {
                'metric_name': metric_name,
                'trend': trend,
                'recent_avg': recent_avg,
                'overall_avg': overall_avg,
                'min_value': min(values),
                'max_value': max(values),
                'data_points': len(values),
                'last_updated': trend_data[-1][0].isoformat() if trend_data else None
            }
    
    def get_system_health_score(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        with self._metrics_lock:
            if not self.current_metrics:
                return {'score': 0, 'status': 'no_data'}
            
            critical_alerts = sum(1 for alert in self.active_alerts.values() 
                                if not alert.resolved and alert.level == AlertLevel.CRITICAL)
            warning_alerts = sum(1 for alert in self.active_alerts.values() 
                               if not alert.resolved and alert.level == AlertLevel.WARNING)
            
            # Health score calculation (0-100)
            base_score = 100
            base_score -= critical_alerts * 30  # -30 per critical alert
            base_score -= warning_alerts * 10   # -10 per warning alert
            
            # Specific metric penalties
            key_metrics = ['system.cpu.usage', 'system.memory.usage']
            for metric_name in key_metrics:
                if metric_name in self.current_metrics:
                    value = self.current_metrics[metric_name].metric_value
                    if value > 90:
                        base_score -= 20
                    elif value > 80:
                        base_score -= 10
                    elif value > 70:
                        base_score -= 5
            
            health_score = max(0, min(100, base_score))
            
            if health_score >= 90:
                status = 'excellent'
            elif health_score >= 70:
                status = 'good'
            elif health_score >= 50:
                status = 'fair'
            elif health_score >= 30:
                status = 'poor'
            else:
                status = 'critical'
            
            return {
                'score': health_score,
                'status': status,
                'critical_alerts': critical_alerts,
                'warning_alerts': warning_alerts,
                'last_updated': datetime.now().isoformat()
            }


def test_metrics_collector():
    """Test metrics collector functionality"""
    
    print("📊 TESTING METRICS COLLECTOR")
    print("=" * 60)
    
    # Create metrics collector
    collector = MetricsCollector({
        'collection_interval': 2,  # 2 seconds for testing
        'alert_cooldown': 10
    })
    
    print("✅ Metrics collector created")
    
    # Add alert callback
    def alert_callback(alert: Alert):
        print(f"🚨 ALERT: {alert.message}")
    
    collector.add_alert_callback(alert_callback)
    print("✅ Alert callback registered")
    
    async def run_test():
        # Start collection
        await collector.start_collection()
        print("✅ Collection started")
        
        # Wait for a few collection cycles
        await asyncio.sleep(8)
        
        # Get current metrics
        current = collector.get_current_metrics()
        print(f"\n📈 Current metrics ({len(current['metrics'])} metrics):")
        
        for name, metric in current['metrics'].items():
            print(f"   {name}: {metric['metric_value']:.2f} {metric['metric_unit']}")
        
        # Get health score
        health = collector.get_system_health_score()
        print(f"\n💚 System health: {health['score']}/100 ({health['status']})")
        
        # Get alerts
        active_alerts = collector.get_active_alerts()
        print(f"\n🚨 Active alerts: {len(active_alerts)}")
        for alert in active_alerts:
            print(f"   {alert['level'].upper()}: {alert['message']}")
        
        # Stop collection
        await collector.stop_collection()
        print("✅ Collection stopped")
        
        return collector
    
    # Run async test
    result_collector = asyncio.run(run_test())
    
    print(f"\n✅ METRICS COLLECTOR TEST COMPLETED!")
    return result_collector


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    test_metrics_collector()
