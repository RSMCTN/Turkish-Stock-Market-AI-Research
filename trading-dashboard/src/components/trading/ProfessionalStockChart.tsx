'use client';

import React, { useEffect, useRef } from 'react';
import { registerLicense } from '@syncfusion/ej2-base';
import {
  StockChartComponent,
  StockChartSeriesCollectionDirective,
  StockChartSeriesDirective,
  Inject,
  DateTime,
  Tooltip,
  RangeTooltip,
  Crosshair,
  LineSeries,
  SplineSeries,
  CandleSeries,
  HiloOpenCloseSeries,
  HiloSeries,
  RangeAreaSeries,
  Trendlines,
  EmaIndicator,
  RsiIndicator,
  BollingerBands,
  TmaIndicator,
  MomentumIndicator,
  SmaIndicator,
  AtrIndicator,
  MacdIndicator,
  AdxIndicator,
  StochasticIndicator,
  Export,
  Selection,
  RangeSelector,
  IStockChartEventArgs,
  ChartTheme
} from '@syncfusion/ej2-react-charts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, RefreshCw, Settings } from 'lucide-react';

// Register SyncFusion License
registerLicense('Ngo9BigBOggjHTQxAR8/V1JEaF5cXmRCf1FpRmJGdld5fUVHYVZUTXxaS00DNHVRdkdmWXZecXRdR2VdWUxwW0VWYEk=');

interface StockData {
  date: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface ProfessionalStockChartProps {
  selectedSymbol?: string;
  data?: StockData[];
  height?: string;
}

const ProfessionalStockChart: React.FC<ProfessionalStockChartProps> = ({
  selectedSymbol = 'AKBNK',
  data,
  height = '500px'
}) => {
  const stockchartRef = useRef<StockChartComponent>(null);

  // Generate realistic mock data for demonstration
  const generateMockData = (): StockData[] => {
    const basePrice = {
      'AKBNK': 69.5,
      'BIMAS': 536.0,
      'GARAN': 145.8,
      'THYAO': 340.0,
      'TUPRS': 171.0,
      'ISCTR': 15.14,
      'SISE': 40.74,
      'ARCLK': 141.2,
      'A1YEN': 58.0,
    }[selectedSymbol] || 50.0;

    const mockData: StockData[] = [];
    const startDate = new Date();
    startDate.setMonth(startDate.getMonth() - 6); // 6 months of data

    for (let i = 0; i < 180; i++) {
      const currentDate = new Date(startDate);
      currentDate.setDate(startDate.getDate() + i);
      
      const price = basePrice + (Math.sin(i * 0.1) * basePrice * 0.1) + (Math.random() - 0.5) * basePrice * 0.05;
      const open = price * (0.995 + Math.random() * 0.01);
      const close = price * (0.995 + Math.random() * 0.01);
      const high = Math.max(open, close) * (1 + Math.random() * 0.02);
      const low = Math.min(open, close) * (1 - Math.random() * 0.02);
      const volume = Math.floor(Math.random() * 10000000) + 1000000;

      mockData.push({
        date: currentDate,
        open: parseFloat(open.toFixed(2)),
        high: parseFloat(high.toFixed(2)),
        low: parseFloat(low.toFixed(2)),
        close: parseFloat(close.toFixed(2)),
        volume: volume
      });
    }

    return mockData;
  };

  const chartData = data || generateMockData();

  const onLoad = (args: IStockChartEventArgs) => {
    let selectedTheme: string = location.hash.split('/')[1];
    selectedTheme = selectedTheme ? selectedTheme : 'Material';
    args.stockChart.theme = (selectedTheme.charAt(0).toUpperCase() + selectedTheme.slice(1)) as ChartTheme;
  };

  return (
    <Card className="w-full bg-gradient-to-br from-slate-900 to-slate-800 border-slate-700">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <TrendingUp className="h-6 w-6 text-emerald-400" />
            <div>
              <CardTitle className="text-slate-100">Bloomberg-Style Stock Chart - {selectedSymbol}</CardTitle>
              <p className="text-sm text-slate-400 mt-1">
                Professional Trading Terminal powered by SyncFusion
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-emerald-400 border-emerald-400">
              {chartData.length} records
            </Badge>
            <Badge className="bg-blue-600 text-white">
              SyncFusion Enterprise
            </Badge>
            <RefreshCw className="h-4 w-4 text-slate-400" />
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="p-4 bg-slate-900/50 rounded-lg border border-slate-700">
          <StockChartComponent
            ref={stockchartRef}
            id="stockchart"
            height={height}
            primaryXAxis={{
              valueType: 'DateTime',
              majorGridLines: { color: 'rgba(255,255,255,0.1)' },
              lineStyle: { color: 'rgba(255,255,255,0.2)' },
              labelStyle: { color: '#9ca3af' }
            }}
            primaryYAxis={{
              labelFormat: '₺{value}',
              majorGridLines: { color: 'rgba(255,255,255,0.1)' },
              lineStyle: { color: 'rgba(255,255,255,0.2)' },
              labelStyle: { color: '#9ca3af' }
            }}
            indicatorType={[]}
            seriesType={[]}
            trendlineType={[]}
            exportType={[]}
            chartArea={{ 
              background: 'transparent',
              border: { color: 'rgba(255,255,255,0.1)', width: 1 }
            }}
            background="transparent"
            theme="MaterialDark"
            tooltip={{ 
              enable: true,
              format: '<b>${point.x}</b><br/>Open: <b>₺${point.open}</b><br/>High: <b>₺${point.high}</b><br/>Low: <b>₺${point.low}</b><br/>Close: <b>₺${point.close}</b><br/>Volume: <b>${point.volume}</b>',
              textStyle: { color: '#ffffff' },
              fill: 'rgba(30, 41, 59, 0.9)',
              border: { color: 'rgba(148, 163, 184, 0.3)', width: 1 }
            }}
            crosshair={{ 
              enable: true,
              lineType: 'Both',
              line: { color: 'rgba(16, 185, 129, 0.8)', width: 1 }
            }}
            enableSelector={true}
            load={onLoad}
          >
            <Inject services={[
              DateTime, Tooltip, RangeTooltip, Crosshair, LineSeries, SplineSeries,
              CandleSeries, HiloOpenCloseSeries, HiloSeries, RangeAreaSeries, Trendlines,
              EmaIndicator, RsiIndicator, BollingerBands, TmaIndicator, MomentumIndicator,
              SmaIndicator, AtrIndicator, MacdIndicator, AdxIndicator, StochasticIndicator,
              Export, Selection, RangeSelector
            ]} />
            <StockChartSeriesCollectionDirective>
              <StockChartSeriesDirective
                dataSource={chartData}
                type="Candle"
                xName="date"
                yName="close"
                high="high"
                low="low"
                open="open"
                close="close"
                volume="volume"
                bearFillColor="#ef4444"
                bullFillColor="#10b981"
                name={selectedSymbol}
              />
            </StockChartSeriesCollectionDirective>
          </StockChartComponent>
        </div>
        
        {/* Quick Stats */}
        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700">
            <div className="text-xs text-slate-400">Current Price</div>
            <div className="text-lg font-bold text-emerald-400">
              ₺{chartData[chartData.length - 1]?.close.toFixed(2)}
            </div>
          </div>
          <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700">
            <div className="text-xs text-slate-400">Daily Change</div>
            <div className="text-lg font-bold text-blue-400">
              {chartData.length > 1 
                ? `₺${(chartData[chartData.length - 1]?.close - chartData[chartData.length - 2]?.close).toFixed(2)}`
                : '₺0.00'
              }
            </div>
          </div>
          <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700">
            <div className="text-xs text-slate-400">Volume</div>
            <div className="text-lg font-bold text-purple-400">
              {chartData[chartData.length - 1]?.volume.toLocaleString('tr-TR')}
            </div>
          </div>
          <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700">
            <div className="text-xs text-slate-400">Range</div>
            <div className="text-lg font-bold text-cyan-400">
              ₺{Math.max(...chartData.map(d => d.high)).toFixed(2)} - ₺{Math.min(...chartData.map(d => d.low)).toFixed(2)}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ProfessionalStockChart;
