'use client';

import { useState } from 'react';
import { RefreshCw, Download, Share2, Bell, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface QuickActionsProps {
  onRefresh?: () => void;
  onExport?: () => void;
  onShare?: () => void;
  className?: string;
}

export default function QuickActions({ 
  onRefresh, 
  onExport, 
  onShare, 
  className = '' 
}: QuickActionsProps) {
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    onRefresh?.();
    setTimeout(() => setIsRefreshing(false), 1500);
  };

  return (
    <Card className={`bg-slate-800/30 border-slate-600 backdrop-blur-md ${className}`}>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-white">Quick Actions</h3>
            <Badge variant="outline" className="text-xs text-slate-300">PRO</Badge>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="text-xs hover:scale-105 transition-all"
            >
              <RefreshCw className={`h-3 w-3 mr-1 ${isRefreshing ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={onExport}
              className="text-xs hover:scale-105 transition-all"
            >
              <Download className="h-3 w-3 mr-1" />
              Export
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={onShare}
              className="text-xs hover:scale-105 transition-all"
            >
              <Share2 className="h-3 w-3 mr-1" />
              Share
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="text-xs hover:scale-105 transition-all"
            >
              <Bell className="h-3 w-3 mr-1" />
              Alerts
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
