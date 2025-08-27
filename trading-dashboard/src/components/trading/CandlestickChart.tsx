'use client';

import React from 'react';
import { ResponsiveContainer, ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

interface CandlestickData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  signal?: 'BUY' | 'SELL' | 'HOLD';
  confidence?: number;
}

interface CandlestickChartProps {
  data: CandlestickData[];
  width?: number;
  height?: number;
}

// Custom Candlestick component
const Candlestick = (props: any) => {
  const { payload, x, y, width, height } = props;
  
  if (!payload || !payload.open || !payload.close || !payload.high || !payload.low) {
    return null;
  }
  
  const { open, close, high, low } = payload;
  const isGreen = close >= open;
  
  // Calculate positions
  const bodyHeight = Math.abs(close - open);
  const bodyY = Math.min(open, close);
  const wickTop = Math.max(high, Math.max(open, close));
  const wickBottom = Math.min(low, Math.min(open, close));
  
  // Colors
  const fillColor = isGreen ? '#10b981' : '#ef4444'; // green-500 : red-500
  const strokeColor = isGreen ? '#059669' : '#dc2626'; // green-600 : red-600
  
  // Calculate actual pixel positions
  const yScale = height / (Math.max(...props.data.map((d: any) => d.high)) - Math.min(...props.data.map((d: any) => d.low)));
  const minPrice = Math.min(...props.data.map((d: any) => d.low));
  
  const candleX = x + width * 0.2;
  const candleWidth = width * 0.6;
  
  // Scale prices to pixel positions
  const scaleY = (price: number) => y + height - ((price - minPrice) * yScale);
  
  return (
    <g>
      {/* High-Low line (wick) */}
      <line
        x1={x + width / 2}
        y1={scaleY(high)}
        x2={x + width / 2}
        y2={scaleY(low)}
        stroke={strokeColor}
        strokeWidth={1}
      />
      
      {/* Open-Close body */}
      <rect
        x={candleX}
        y={scaleY(Math.max(open, close))}
        width={candleWidth}
        height={Math.abs(scaleY(close) - scaleY(open))}
        fill={isGreen ? fillColor : 'white'}
        stroke={strokeColor}
        strokeWidth={1}
      />
      
      {/* Signal indicator */}
      {payload.signal && payload.signal !== 'HOLD' && (
        <circle
          cx={x + width / 2}
          cy={scaleY(high) - 8}
          r={3}
          fill={payload.signal === 'BUY' ? '#10b981' : '#ef4444'}
          stroke="white"
          strokeWidth={1}
        />
      )}
    </g>
  );
};

// Simple Candlestick implementation using rectangles
const SimpleCandlestick = ({ data }: { data: CandlestickData[] }) => {
  const minPrice = Math.min(...data.map(d => d.low));
  const maxPrice = Math.max(...data.map(d => d.high));
  const priceRange = maxPrice - minPrice;
  
  return (
    <div className="relative w-full h-full bg-white rounded-lg overflow-hidden">
      <svg width="100%" height="100%" viewBox="0 0 800 400">
        {/* Grid lines */}
        <defs>
          <pattern id="grid" width="40" height="20" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 20" fill="none" stroke="#e5e7eb" strokeWidth="0.5"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
        
        {/* Y-axis labels */}
        {Array.from({ length: 6 }, (_, i) => {
          const price = minPrice + (priceRange * i / 5);
          const y = 350 - (i * 60);
          return (
            <g key={i}>
              <text x="750" y={y} fontSize="10" fill="#6b7280" textAnchor="middle">
                ₺{price.toFixed(1)}
              </text>
              <line x1="60" y1={y} x2="740" y2={y} stroke="#f3f4f6" strokeWidth="1" />
            </g>
          );
        })}
        
        {/* Candlesticks */}
        {data.slice(-20).map((candle, index) => {
          const x = 70 + (index * 32);
          const candleWidth = 24;
          
          const openY = 350 - ((candle.open - minPrice) / priceRange * 300);
          const closeY = 350 - ((candle.close - minPrice) / priceRange * 300);
          const highY = 350 - ((candle.high - minPrice) / priceRange * 300);
          const lowY = 350 - ((candle.low - minPrice) / priceRange * 300);
          
          const isGreen = candle.close >= candle.open;
          const bodyColor = isGreen ? '#10b981' : '#ef4444';
          const fillColor = isGreen ? '#10b981' : 'white';
          
          return (
            <g key={index}>
              {/* High-Low line */}
              <line
                x1={x + candleWidth/2}
                y1={highY}
                x2={x + candleWidth/2}
                y2={lowY}
                stroke={bodyColor}
                strokeWidth="1"
              />
              
              {/* Open-Close body */}
              <rect
                x={x + 2}
                y={Math.min(openY, closeY)}
                width={candleWidth - 4}
                height={Math.abs(closeY - openY) || 1}
                fill={fillColor}
                stroke={bodyColor}
                strokeWidth="1"
              />
              
              {/* Signal indicator */}
              {candle.signal && candle.signal !== 'HOLD' && (
                <circle
                  cx={x + candleWidth/2}
                  cy={highY - 8}
                  r="2"
                  fill={candle.signal === 'BUY' ? '#10b981' : '#ef4444'}
                  stroke="white"
                  strokeWidth="1"
                />
              )}
              
              {/* Time label */}
              {index % 4 === 0 && (
                <text 
                  x={x + candleWidth/2} 
                  y="380" 
                  fontSize="8" 
                  fill="#6b7280" 
                  textAnchor="middle"
                >
                  {candle.time}
                </text>
              )}
            </g>
          );
        })}
        
        {/* Chart title */}
        <text x="400" y="25" fontSize="14" fill="#374151" textAnchor="middle" fontWeight="bold">
          Candlestick Chart - Last 20 Periods
        </text>
      </svg>
    </div>
  );
};

const CandlestickTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-white border-2 border-gray-200 rounded-lg shadow-xl p-3">
        <p className="font-semibold text-gray-900">{label}</p>
        <div className="space-y-1 text-sm">
          <p><span className="font-medium">Open:</span> ₺{data.open?.toFixed(2)}</p>
          <p><span className="font-medium">High:</span> ₺{data.high?.toFixed(2)}</p>
          <p><span className="font-medium">Low:</span> ₺{data.low?.toFixed(2)}</p>
          <p><span className="font-medium">Close:</span> ₺{data.close?.toFixed(2)}</p>
          <p><span className="font-medium">Volume:</span> {data.volume?.toLocaleString()}</p>
          {data.signal && data.signal !== 'HOLD' && (
            <div className="mt-2 pt-2 border-t border-gray-200">
              <span className={`text-xs font-semibold px-2 py-1 rounded ${
                data.signal === 'BUY' 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                {data.signal} {(data.confidence * 100).toFixed(0)}%
              </span>
            </div>
          )}
        </div>
      </div>
    );
  }
  return null;
};

export default function CandlestickChart({ data, width, height = 400 }: CandlestickChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="w-full h-96 flex items-center justify-center bg-gray-50 rounded-lg">
        <p className="text-gray-500">No candlestick data available</p>
      </div>
    );
  }

  return (
    <div style={{ width: width || '100%', height }}>
      <SimpleCandlestick data={data} />
    </div>
  );
}
