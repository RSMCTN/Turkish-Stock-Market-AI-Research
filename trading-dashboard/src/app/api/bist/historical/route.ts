import { NextRequest, NextResponse } from 'next/server';
import { createMatriksClient, generateMockHistoricalData } from '@/lib/matriks-client';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol');
    const period = searchParams.get('period') || '1d';
    const startDate = searchParams.get('start_date');
    const endDate = searchParams.get('end_date');
    const limit = searchParams.get('limit') || '1000';
    const useMock = searchParams.get('mock') === 'true';

    if (!symbol) {
      return NextResponse.json({
        success: false,
        error: 'Symbol parameter is required'
      }, { status: 400 });
    }

    let historicalData;

    if (useMock || !process.env.MATRIKS_API_KEY) {
      console.log(`Using mock historical data for ${symbol}`);
      const days = parseInt(limit) || 30;
      historicalData = generateMockHistoricalData(symbol.toUpperCase(), days);
    } else {
      // Use real MatriksIQ API
      const client = createMatriksClient(process.env.MATRIKS_API_KEY);
      historicalData = await client.getHistoricalData(
        symbol,
        period,
        startDate || undefined,
        endDate || undefined,
        parseInt(limit)
      );

      if (!historicalData || historicalData.length === 0) {
        console.warn(`No historical data from API for ${symbol}, falling back to mock`);
        const days = parseInt(limit) || 30;
        historicalData = generateMockHistoricalData(symbol.toUpperCase(), days);
      }
    }

    // Calculate additional metrics
    const prices = historicalData.map(d => d.close);
    const volumes = historicalData.map(d => d.volume);
    
    const currentPrice = prices[prices.length - 1];
    const prevPrice = prices[prices.length - 2];
    const change = currentPrice - prevPrice;
    const changePercent = (change / prevPrice) * 100;
    
    const high52Week = Math.max(...prices);
    const low52Week = Math.min(...prices);
    const avgVolume = volumes.reduce((sum, v) => sum + v, 0) / volumes.length;

    // Technical indicators (basic calculations)
    const sma20 = calculateSMA(prices, 20);
    const rsi14 = calculateRSI(prices, 14);

    return NextResponse.json({
      success: true,
      data: {
        symbol: symbol.toUpperCase(),
        period,
        data: historicalData,
        summary: {
          current_price: Number(currentPrice?.toFixed(2)) || 0,
          change: Number(change?.toFixed(2)) || 0,
          change_percent: Number(changePercent?.toFixed(2)) || 0,
          high_52w: Number(high52Week?.toFixed(2)) || 0,
          low_52w: Number(low52Week?.toFixed(2)) || 0,
          avg_volume: Math.round(avgVolume) || 0,
          data_points: historicalData.length
        },
        indicators: {
          sma_20: Number(sma20?.toFixed(2)) || 0,
          rsi_14: Number(rsi14?.toFixed(2)) || 50
        }
      },
      meta: {
        timestamp: new Date().toISOString(),
        source: useMock || !process.env.MATRIKS_API_KEY ? 'mock' : 'matriks',
        params: { symbol, period, startDate, endDate, limit }
      }
    });

  } catch (error) {
    console.error('Error fetching historical data:', error);
    
    return NextResponse.json({
      success: false,
      error: 'Failed to fetch historical data',
      message: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

// Simple Moving Average calculation
function calculateSMA(prices: number[], period: number): number | null {
  if (prices.length < period) return null;
  
  const recentPrices = prices.slice(-period);
  const sum = recentPrices.reduce((acc, price) => acc + price, 0);
  return sum / period;
}

// RSI calculation (simplified)
function calculateRSI(prices: number[], period: number = 14): number | null {
  if (prices.length < period + 1) return null;

  const changes: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    changes.push(prices[i] - prices[i - 1]);
  }

  if (changes.length < period) return null;

  const recentChanges = changes.slice(-period);
  const gains = recentChanges.filter(change => change > 0);
  const losses = recentChanges.filter(change => change < 0).map(loss => Math.abs(loss));

  const avgGain = gains.length > 0 ? gains.reduce((sum, gain) => sum + gain, 0) / period : 0;
  const avgLoss = losses.length > 0 ? losses.reduce((sum, loss) => sum + loss, 0) / period : 0;

  if (avgLoss === 0) return 100;
  
  const rs = avgGain / avgLoss;
  const rsi = 100 - (100 / (1 + rs));
  
  return Math.max(0, Math.min(100, rsi));
}

// Handle OPTIONS for CORS
export async function OPTIONS() {
  return NextResponse.json(
    {},
    {
      status: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      },
    }
  );
}
