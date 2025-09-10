'use client';

import { useEffect, useRef } from 'react';

interface TradingViewMarketOverviewProps {
  market: string;
}

export function TradingViewMarketOverview({ market }: TradingViewMarketOverviewProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      // Clear previous widget
      containerRef.current.innerHTML = '';
      
      const script = document.createElement('script');
      script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js';
      script.type = 'text/javascript';
      script.async = true;
      
      // Configure widget based on selected market
      const config = getMarketConfig(market);
      script.innerHTML = JSON.stringify(config);
      
      containerRef.current.appendChild(script);
    }
  }, [market]);

  return (
    <div className="tradingview-widget-container" style={{ height: '400px', width: '100%' }}>
      <div 
        className="tradingview-widget-container__widget" 
        ref={containerRef}
        style={{ height: '100%', width: '100%' }}
      />
      <div className="tradingview-widget-copyright">
        <a 
          href="https://www.tradingview.com/" 
          rel="noopener nofollow" 
          target="_blank"
          className="text-blue-500 text-xs"
        >
          Track all markets on TradingView
        </a>
      </div>
    </div>
  );
}

function getMarketConfig(market: string) {
  const baseConfig = {
    "colorTheme": "dark",
    "dateRange": "12M",
    "showChart": true,
    "locale": "tr",
    "width": "100%",
    "height": "100%",
    "largeChartUrl": "",
    "isTransparent": true,
    "showSymbolLogo": true,
    "showFloatingTooltip": false,
    "plotLineColorGrowing": "rgba(34, 197, 94, 1)",
    "plotLineColorFalling": "rgba(239, 68, 68, 1)",
    "gridLineColor": "rgba(30, 41, 59, 0.1)",
    "scaleFontColor": "rgba(148, 163, 184, 1)",
    "belowLineFillColorGrowing": "rgba(34, 197, 94, 0.1)",
    "belowLineFillColorFalling": "rgba(239, 68, 68, 0.1)",
    "belowLineFillColorGrowingBottom": "rgba(34, 197, 94, 0)",
    "belowLineFillColorFallingBottom": "rgba(239, 68, 68, 0)",
    "symbolActiveColor": "rgba(59, 130, 246, 0.2)"
  };

  switch (market) {
    case 'turkey':
      return {
        ...baseConfig,
        "tabs": [
          {
            "title": "BIST 30",
            "symbols": [
              {"s": "BIST:XU030", "d": "BIST 30"},
              {"s": "BIST:AKBNK", "d": "Akbank"},
              {"s": "BIST:GARAN", "d": "Garanti"}, 
              {"s": "BIST:ISCTR", "d": "İş Bankası"},
              {"s": "BIST:YKBNK", "d": "Yapı Kredi"},
              {"s": "BIST:VAKBN", "d": "VakıfBank"},
              {"s": "BIST:TUPRS", "d": "Tüpraş"},
              {"s": "BIST:ASELS", "d": "Aselsan"},
              {"s": "BIST:TCELL", "d": "Turkcell"},
              {"s": "BIST:THYAO", "d": "THY"},
              {"s": "BIST:KCHOL", "d": "Koç Holding"},
              {"s": "BIST:SAHOL", "d": "Sabancı"}
            ],
            "originalTitle": "BIST"
          }
        ]
      };
      
    case 'usa':
      return {
        ...baseConfig,
        "tabs": [
          {
            "title": "US Markets",
            "symbols": [
              {"s": "NASDAQ:AAPL"},
              {"s": "NASDAQ:MSFT"},
              {"s": "NASDAQ:GOOGL"},
              {"s": "NASDAQ:AMZN"},
              {"s": "NASDAQ:TSLA"},
              {"s": "NASDAQ:META"},
              {"s": "NASDAQ:NVDA"},
              {"s": "NYSE:JPM"},
              {"s": "NYSE:JNJ"},
              {"s": "NYSE:V"}
            ],
            "originalTitle": "US"
          }
        ]
      };
      
    case 'global':
      return {
        ...baseConfig,
        "tabs": [
          {
            "title": "Global Mix",
            "symbols": [
              {"s": "BIST:XU100", "d": "Turkey"},
              {"s": "NASDAQ:QQQ", "d": "USA"},
              {"s": "LSE:FTSE", "d": "UK"},
              {"s": "XETR:DAX", "d": "Germany"},
              {"s": "TSE:N225", "d": "Japan"},
              {"s": "HKEX:HSI", "d": "Hong Kong"},
              {"s": "NSE:NIFTY", "d": "India"},
              {"s": "SSE:000001", "d": "China"},
              {"s": "TSX:TCOMP", "d": "Canada"},
              {"s": "ASX:XAO", "d": "Australia"}
            ],
            "originalTitle": "Global"
          }
        ]
      };
      
    default:
      return {
        ...baseConfig,
        "tabs": [
          {
            "title": "Major Markets",
            "symbols": [
              {"s": "BIST:XU100"},
              {"s": "SPX:SPX"},
              {"s": "NASDAQ:QQQ"},
              {"s": "XETR:DAX"},
              {"s": "LSE:FTSE"}
            ],
            "originalTitle": "Indices"
          }
        ]
      };
  }
}
