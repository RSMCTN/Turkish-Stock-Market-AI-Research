import { NextResponse } from 'next/server';

// Auth0 temporarily disabled for development
export async function GET() {
  return NextResponse.json({
    message: 'Auth0 temporarily disabled for development',
    status: 'disabled'
  });
}
