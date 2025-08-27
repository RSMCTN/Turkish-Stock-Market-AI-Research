import { NextRequest, NextResponse } from "next/server";
import { checkRedisHealth, bistCache, getRedis } from "@/lib/redis";

export async function GET() {
  const testResults = {
    redis_health: false,
    cache_write: false,
    cache_read: false,
    connection_info: {},
    environment: {},
    errors: [] as string[]
  };

  try {
    // Test 1: Basic Redis health check
    console.log('🔍 Testing Redis health...');
    testResults.redis_health = await checkRedisHealth();
    
    // Test 2: Test cache write operation
    console.log('🔍 Testing cache write...');
    try {
      await bistCache.cacheSymbols([
        { symbol: "TEST", name: "Test Symbol", sector: "Test" },
        { symbol: "DEMO", name: "Demo Symbol", sector: "Demo" }
      ], 60);
      testResults.cache_write = true;
    } catch (error) {
      testResults.errors.push(`Cache write failed: ${error instanceof Error ? error.message : 'Unknown'}`);
    }

    // Test 3: Test cache read operation
    console.log('🔍 Testing cache read...');
    try {
      const cached = await bistCache.getCachedSymbols();
      testResults.cache_read = cached !== null && Array.isArray(cached) && cached.length > 0;
    } catch (error) {
      testResults.errors.push(`Cache read failed: ${error instanceof Error ? error.message : 'Unknown'}`);
    }

    // Test 4: Direct Redis ping test
    console.log('🔍 Testing direct Redis ping...');
    try {
      const redis = getRedis();
      const pingResult = await redis.ping();
      testResults.connection_info = {
        ping: pingResult,
        status: redis.status,
        host: process.env.REDIS_HOST,
        port: process.env.REDIS_PORT,
        has_password: !!process.env.REDIS_PASSWORD
      };
    } catch (error) {
      testResults.errors.push(`Direct ping failed: ${error instanceof Error ? error.message : 'Unknown'}`);
    }

    // Environment info
    testResults.environment = {
      node_env: process.env.NODE_ENV,
      redis_url_available: !!process.env.REDIS_URL,
      redis_host_available: !!process.env.REDIS_HOST,
      redis_password_available: !!process.env.REDIS_PASSWORD
    };

    const overallSuccess = testResults.redis_health && testResults.cache_write && testResults.cache_read;

    return NextResponse.json({
      success: overallSuccess,
      message: overallSuccess ? "All Redis tests passed!" : "Some Redis tests failed",
      tests: testResults,
      timestamp: new Date().toISOString()
    }, { status: overallSuccess ? 200 : 500 });

  } catch (error) {
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
      tests: testResults,
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
}
