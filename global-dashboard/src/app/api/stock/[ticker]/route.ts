import { NextRequest, NextResponse } from 'next/server';

const PROFIT_API_KEY = "a9a0bacbab08493d958244c05380da01";
const PROFIT_BASE_URL = "https://api.profit.com";

export async function GET(
  request: NextRequest,
  { params }: { params: { ticker: string } }
) {
  try {
    const { ticker } = params;
    
    // Validate ticker format
    if (!ticker || ticker.trim() === '') {
      return NextResponse.json(
        { error: 'Ticker parameter required' },
        { status: 400 }
      );
    }

    // Ensure ticker has .IS suffix for Turkish stocks
    const formattedTicker = ticker.endsWith('.IS') ? ticker : `${ticker}.IS`;
    
    // Call Profit.com API
    const apiUrl = `${PROFIT_BASE_URL}/data-api/market-data/quote/${formattedTicker}?token=${PROFIT_API_KEY}`;
    
    console.log(`🔍 Fetching data from: ${apiUrl}`);
    
    const response = await fetch(apiUrl, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'MAMUT-R600-Dashboard/1.0'
      },
      // Add timeout
      signal: AbortSignal.timeout(10000) // 10 seconds
    });

    if (!response.ok) {
      console.error(`❌ Profit API error: ${response.status} ${response.statusText}`);
      
      // Return structured error
      return NextResponse.json(
        { 
          error: `API request failed: ${response.status}`,
          ticker: formattedTicker,
          timestamp: Date.now()
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    
    console.log(`✅ Successfully fetched data for ${formattedTicker}`);
    
    // Add metadata
    const enrichedData = {
      ...data,
      api_source: 'Profit.com',
      fetch_timestamp: Date.now(),
      requested_ticker: ticker,
      formatted_ticker: formattedTicker
    };

    return NextResponse.json(enrichedData);

  } catch (error: any) {
    console.error('❌ API route error:', error);
    
    // Handle timeout and network errors
    if (error.name === 'TimeoutError' || error.code === 'ECONNRESET') {
      return NextResponse.json(
        { 
          error: 'API request timeout',
          message: 'Profit.com API took too long to respond',
          timestamp: Date.now()
        },
        { status: 408 }
      );
    }

    return NextResponse.json(
      { 
        error: 'Internal server error',
        message: error.message || 'Unknown error occurred',
        timestamp: Date.now()
      },
      { status: 500 }
    );
  }
}
