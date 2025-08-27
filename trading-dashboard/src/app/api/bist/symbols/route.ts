import { NextRequest, NextResponse } from 'next/server';
import { createMatriksClient, generateMockSymbols } from '@/lib/matriks-client';
import { bistCache, rateLimiter } from '@/lib/redis';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const useMock = searchParams.get('mock') === 'true';
    const sector = searchParams.get('sector');
    const limit = searchParams.get('limit');
    const forceRefresh = searchParams.get('refresh') === 'true';

    // Rate limiting check
    const clientIp = request.headers.get('x-forwarded-for') || 'anonymous';
    const rateCheck = await rateLimiter.checkRateLimit(`symbols:${clientIp}`, 60, 3600);
    
    if (!rateCheck.allowed) {
      return NextResponse.json({
        success: false,
        error: 'Rate limit exceeded',
        resetTime: rateCheck.resetTime
      }, { 
        status: 429,
        headers: {
          'X-RateLimit-Remaining': rateCheck.remaining.toString(),
          'X-RateLimit-Reset': rateCheck.resetTime.toString()
        }
      });
    }

    let symbols;

    // Try cache first (unless force refresh or using mock)
    if (!forceRefresh && !useMock && process.env.REDIS_URL) {
      try {
        symbols = await bistCache.getCachedSymbols();
        if (symbols) {
          console.log(`✅ Symbols loaded from cache: ${symbols.length} items`);
        }
      } catch (error) {
        console.warn('Redis cache error:', error);
      }
    }

    // Fetch from API if not cached
    if (!symbols) {
      if (useMock || !process.env.MATRIKS_API_KEY) {
        console.log('Using mock BIST symbols data');
        symbols = generateMockSymbols();
      } else {
        // Use real MatriksIQ API
        const client = createMatriksClient(process.env.MATRIKS_API_KEY);
        symbols = await client.getBistSymbols();
        
        if (!symbols || symbols.length === 0) {
          console.warn('No symbols from API, falling back to mock data');
          symbols = generateMockSymbols();
        }
      }

      // Cache the results
      if (symbols && process.env.REDIS_URL) {
        try {
          await bistCache.cacheSymbols(symbols, 3600); // Cache for 1 hour
          console.log(`✅ Symbols cached: ${symbols.length} items`);
        } catch (error) {
          console.warn('Failed to cache symbols:', error);
        }
      }
    }

    // Apply filters
    if (sector) {
      symbols = symbols.filter(s => 
        s.sector.toLowerCase().includes(sector.toLowerCase())
      );
    }

    // Apply limit
    if (limit) {
      const limitNum = parseInt(limit);
      if (limitNum > 0) {
        symbols = symbols.slice(0, limitNum);
      }
    }

    // Group by sector for better organization
    const groupedBySector = symbols.reduce((acc, symbol) => {
      const sectorKey = symbol.sector || 'Other';
      if (!acc[sectorKey]) {
        acc[sectorKey] = [];
      }
      acc[sectorKey].push(symbol);
      return acc;
    }, {} as Record<string, typeof symbols>);

    return NextResponse.json({
      success: true,
      data: {
        symbols,
        total: symbols.length,
        sectors: Object.keys(groupedBySector).length,
        grouped: groupedBySector
      },
      meta: {
        timestamp: new Date().toISOString(),
        source: useMock || !process.env.MATRIKS_API_KEY ? 'mock' : 'matriks',
        filters: { sector, limit }
      }
    });

  } catch (error) {
    console.error('Error fetching BIST symbols:', error);
    
    return NextResponse.json({
      success: false,
      error: 'Failed to fetch BIST symbols',
      message: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
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
