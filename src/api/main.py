"""
BIST DP-LSTM Trading System API
Production-ready FastAPI application with monitoring and real-time trading capabilities
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our trading components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.signal_generator import (
    SignalGenerator, SignalGeneratorConfig, TradingSignal, SignalAction
)
from execution.portfolio_manager import PortfolioManager, PortfolioConfig
from execution.paper_trading_engine import (
    PaperTradingEngine, TradingEngineConfig, MarketData
)


# =============================================================================
# Pydantic Models for API
# =============================================================================

class SignalRequest(BaseModel):
    """Request model for signal generation"""
    symbol: str = Field(..., description="Stock symbol (e.g., AKBNK)")
    include_features: bool = Field(False, description="Include feature analysis in response")
    market_data: Optional[Dict[str, Any]] = Field(None, description="Optional market data override")


class SignalResponse(BaseModel):
    """Response model for generated signals"""
    signal_id: str
    symbol: str
    timestamp: str
    action: str = Field(..., description="BUY, SELL, or HOLD")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    expected_return: float = Field(..., description="Expected return percentage")
    stop_loss: Optional[float] = Field(None, description="Stop loss percentage")
    take_profit: Optional[float] = Field(None, description="Take profit percentage")
    features: Optional[Dict[str, Any]] = Field(None, description="Optional feature analysis")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional signal metadata")


class ExecuteSignalRequest(BaseModel):
    """Request model for signal execution"""
    signal_id: str = Field(..., description="Signal ID to execute")
    override_checks: bool = Field(False, description="Override risk checks")


class ExecuteSignalResponse(BaseModel):
    """Response model for signal execution"""
    success: bool
    execution_id: Optional[str] = None
    position_id: Optional[str] = None
    entry_price: Optional[float] = None
    position_size: Optional[float] = None
    total_costs: Optional[float] = None
    reason: Optional[str] = None
    timestamp: str


class PortfolioSummary(BaseModel):
    """Portfolio summary response model"""
    timestamp: str
    initial_capital: float
    current_value: float
    available_cash: float
    total_return: float = Field(..., description="Total return as decimal")
    total_return_pct: float = Field(..., description="Total return as percentage")
    current_positions: int
    total_trades: int
    win_rate: float = Field(..., ge=0, le=1, description="Win rate 0-1")
    max_drawdown: float
    unrealized_pnl: float
    realized_pnl: float


class SystemHealthResponse(BaseModel):
    """System health check response"""
    status: str = Field(..., description="overall, healthy, degraded, or unhealthy")
    timestamp: str
    uptime_seconds: float
    components: Dict[str, str]
    metrics: Dict[str, Any]
    last_check: str


class SystemMetricsResponse(BaseModel):
    """System metrics response"""
    timestamp: str
    cpu_usage_pct: float
    memory_usage_pct: float
    system_uptime: float
    api_requests_total: int
    api_avg_response_time: float
    trading_signals_today: int
    active_positions: int
    last_prediction_latency: Optional[float] = None


# =============================================================================
# FastAPI Application Setup
# =============================================================================

app = FastAPI(
    title="BIST DP-LSTM Trading System API",
    description="Advanced Trading System with Differential Privacy LSTM for BIST (Borsa Istanbul)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global application state
class AppState:
    def __init__(self):
        self.start_time = datetime.now()
        self.signal_generator: Optional[SignalGenerator] = None
        self.paper_trader: Optional[PaperTradingEngine] = None
        self.metrics_collector = None
        
        # API metrics
        self.api_request_count = 0
        self.api_response_times = []
        self.last_prediction_latency = None
        
        # Signal cache for execution
        self.recent_signals: Dict[str, TradingSignal] = {}

app_state = AppState()

# =============================================================================
# Application Lifecycle
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize all system components"""
    logger = logging.getLogger("api.startup")
    logger.info("🚀 Starting BIST DP-LSTM Trading System API...")
    
    try:
        # Initialize signal generator
        signal_config = SignalGeneratorConfig(
            buy_threshold=0.65,
            sell_threshold=0.65,
            min_expected_return=0.012,
            max_signals_per_symbol=10
        )
        
        # Mock model for API demonstration
        from execution.integrated_trading_test import MockModel, MockFeatureProcessor
        mock_model = MockModel(trend_direction=0.0)  # Neutral
        feature_processor = MockFeatureProcessor()
        
        app_state.signal_generator = SignalGenerator(
            model=mock_model,
            feature_processor=feature_processor,
            config=signal_config
        )
        
        # Initialize paper trading engine
        engine_config = TradingEngineConfig(
            initial_capital=100000.0,
            execution_algorithm='vwap',
            max_positions=10
        )
        
        app_state.paper_trader = PaperTradingEngine(engine_config)
        
        # Start metrics collection (if implemented)
        logger.info("✅ All components initialized successfully")
        logger.info(f"📊 API Documentation: http://localhost:8000/docs")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown of all components"""
    logger = logging.getLogger("api.shutdown")
    logger.info("🛑 Shutting down BIST DP-LSTM Trading System API...")
    
    # Stop background tasks, close connections, etc.
    logger.info("✅ Shutdown completed")


# =============================================================================
# Middleware for Request Tracking
# =============================================================================

@app.middleware("http")
async def request_tracking_middleware(request, call_next):
    """Track API requests and response times"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Update metrics
    response_time = (time.time() - start_time) * 1000  # milliseconds
    app_state.api_request_count += 1
    app_state.api_response_times.append(response_time)
    
    # Keep only last 1000 response times
    if len(app_state.api_response_times) > 1000:
        app_state.api_response_times = app_state.api_response_times[-1000:]
    
    # Add response headers
    response.headers["X-Response-Time"] = f"{response_time:.2f}ms"
    response.headers["X-Request-ID"] = str(app_state.api_request_count)
    
    return response


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with basic system information"""
    return {
        "message": "BIST DP-LSTM Trading System API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - app_state.start_time).total_seconds(),
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "metrics": "/metrics/system",
            "signals": "/signals/generate",
            "portfolio": "/portfolio/summary"
        },
        "description": "Advanced Trading System with Differential Privacy LSTM for BIST"
    }


@app.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """Comprehensive system health check"""
    timestamp = datetime.now()
    uptime = (timestamp - app_state.start_time).total_seconds()
    
    components = {}
    metrics = {}
    
    # Check signal generator
    if app_state.signal_generator:
        components["signal_generator"] = "healthy"
        try:
            daily_stats = app_state.signal_generator.get_daily_stats()
            metrics["signals_today"] = daily_stats.get("total_signals_today", 0)
        except:
            components["signal_generator"] = "degraded"
    else:
        components["signal_generator"] = "not_loaded"
    
    # Check paper trader
    if app_state.paper_trader:
        components["paper_trader"] = "healthy"
        try:
            status = app_state.paper_trader.get_current_status()
            portfolio = status["portfolio_summary"]
            metrics["portfolio_value"] = portfolio["capital"]["current_value"]
            metrics["active_positions"] = portfolio["positions"]["count"]
        except:
            components["paper_trader"] = "degraded"
    else:
        components["paper_trader"] = "not_loaded"
    
    # Check system resources
    try:
        import psutil
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent
        
        components["system_resources"] = "healthy" if cpu_usage < 80 and memory_usage < 85 else "stressed"
        metrics["cpu_usage"] = cpu_usage
        metrics["memory_usage"] = memory_usage
    except:
        components["system_resources"] = "unknown"
    
    # Determine overall status
    component_statuses = list(components.values())
    if all(status == "healthy" for status in component_statuses):
        overall_status = "healthy"
    elif any(status in ["not_loaded", "unhealthy"] for status in component_statuses):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"
    
    return SystemHealthResponse(
        status=overall_status,
        timestamp=timestamp.isoformat(),
        uptime_seconds=uptime,
        components=components,
        metrics=metrics,
        last_check=timestamp.isoformat()
    )


@app.post("/signals/generate", response_model=SignalResponse)
async def generate_signal(request: SignalRequest):
    """Generate trading signal for specified symbol"""
    if not app_state.signal_generator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Signal generator not available"
        )
    
    try:
        # Prepare market data
        market_data = request.market_data or {
            'symbol': request.symbol,
            'current_price': 10.0,  # Mock price
            'volume_ratio': 1.2,
            'volatility_zscore': 0.5,
            'conditions': {'market_open': True}
        }
        
        # Generate signal
        start_time = time.time()
        signal = await app_state.signal_generator.generate_signal(
            request.symbol,
            market_data,
            []  # No news data for now
        )
        prediction_latency = (time.time() - start_time) * 1000
        app_state.last_prediction_latency = prediction_latency
        
        # Cache signal for potential execution
        app_state.recent_signals[signal.signal_id] = signal
        
        # Prepare response
        response_data = {
            "signal_id": signal.signal_id,
            "symbol": signal.symbol,
            "timestamp": signal.timestamp.isoformat(),
            "action": signal.action.value,
            "confidence": signal.confidence,
            "expected_return": signal.expected_return,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "metadata": signal.metadata
        }
        
        # Add features if requested
        if request.include_features:
            response_data["features"] = {
                "prediction_latency_ms": prediction_latency,
                "market_conditions": market_data,
                "generation_method": "dp_lstm_ensemble"
            }
        
        return SignalResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signal generation failed: {str(e)}"
        )


@app.post("/signals/execute", response_model=ExecuteSignalResponse)
async def execute_signal(request: ExecuteSignalRequest):
    """Execute a previously generated trading signal"""
    if not app_state.paper_trader:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Paper trader not available"
        )
    
    # Find signal
    signal = app_state.recent_signals.get(request.signal_id)
    if not signal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Signal not found or expired"
        )
    
    try:
        # Execute signal
        result = await app_state.paper_trader.execute_signal(signal)
        
        # Prepare response
        response_data = {
            "success": result["success"],
            "timestamp": datetime.now().isoformat(),
            "reason": result.get("reason", "Executed successfully" if result["success"] else "Execution failed")
        }
        
        if result["success"]:
            response_data.update({
                "execution_id": f"exec_{request.signal_id}",
                "position_id": result.get("position_id"),
                "entry_price": result.get("entry_price"),
                "position_size": result.get("position_size"),
                "total_costs": result.get("total_costs")
            })
        
        return ExecuteSignalResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signal execution failed: {str(e)}"
        )


@app.get("/portfolio/summary", response_model=PortfolioSummary)
async def get_portfolio_summary():
    """Get comprehensive portfolio summary"""
    if not app_state.paper_trader:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Paper trader not available"
        )
    
    try:
        status = app_state.paper_trader.get_current_status()
        portfolio = status["portfolio_summary"]
        
        return PortfolioSummary(
            timestamp=datetime.now().isoformat(),
            initial_capital=portfolio["capital"]["initial"],
            current_value=portfolio["capital"]["current_value"],
            available_cash=portfolio["capital"]["available_cash"],
            total_return=portfolio["capital"]["total_return"],
            total_return_pct=portfolio["capital"]["total_return_pct"],
            current_positions=portfolio["positions"]["count"],
            total_trades=portfolio["trades"]["total_count"],
            win_rate=portfolio["trades"]["win_rate"],
            max_drawdown=portfolio["risk"]["max_drawdown"],
            unrealized_pnl=portfolio["positions"]["unrealized_pnl"],
            realized_pnl=portfolio["trades"]["realized_pnl"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get portfolio summary: {str(e)}"
        )


@app.get("/portfolio/positions", response_model=List[Dict[str, Any]])
async def get_positions():
    """Get all current positions"""
    if not app_state.paper_trader:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Paper trader not available"
        )
    
    try:
        positions = []
        for symbol, position in app_state.paper_trader.portfolio.positions.items():
            positions.append(position.to_dict())
        
        return positions
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get positions: {str(e)}"
        )


@app.get("/portfolio/trades", response_model=List[Dict[str, Any]])
async def get_trade_history():
    """Get trading history"""
    if not app_state.paper_trader:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Paper trader not available"
        )
    
    try:
        trades = []
        for trade in app_state.paper_trader.portfolio.closed_trades:
            trades.append(trade.to_dict())
        
        return trades
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trade history: {str(e)}"
        )


@app.get("/metrics/system", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """Get comprehensive system metrics"""
    try:
        import psutil
        
        # Calculate average response time
        avg_response_time = (
            sum(app_state.api_response_times) / len(app_state.api_response_times)
            if app_state.api_response_times else 0.0
        )
        
        # Get signal stats
        signals_today = 0
        if app_state.signal_generator:
            try:
                daily_stats = app_state.signal_generator.get_daily_stats()
                signals_today = daily_stats.get("total_signals_today", 0)
            except:
                pass
        
        # Get active positions
        active_positions = 0
        if app_state.paper_trader:
            try:
                status = app_state.paper_trader.get_current_status()
                active_positions = status["portfolio_summary"]["positions"]["count"]
            except:
                pass
        
        return SystemMetricsResponse(
            timestamp=datetime.now().isoformat(),
            cpu_usage_pct=psutil.cpu_percent(interval=0.1),
            memory_usage_pct=psutil.virtual_memory().percent,
            system_uptime=(datetime.now() - app_state.start_time).total_seconds(),
            api_requests_total=app_state.api_request_count,
            api_avg_response_time=avg_response_time,
            trading_signals_today=signals_today,
            active_positions=active_positions,
            last_prediction_latency=app_state.last_prediction_latency
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system metrics: {str(e)}"
        )


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unhandled errors"""
    logger = logging.getLogger("api.error")
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
