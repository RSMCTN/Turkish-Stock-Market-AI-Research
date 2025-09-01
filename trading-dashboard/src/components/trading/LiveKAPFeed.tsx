'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertCircle } from 'lucide-react';

interface LiveKAPFeedProps {
  selectedSymbol?: string;
}

export default function LiveKAPFeed({ selectedSymbol = 'GARAN' }: LiveKAPFeedProps) {
  return (
    <Card className="bg-gradient-to-br from-green-50 to-blue-50">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertCircle className="h-5 w-5 text-green-600" />
          Canlı KAP Beslemesi - {selectedSymbol}
        </CardTitle>
        <Badge className="w-fit bg-green-100 text-green-700">Gerçek-zamanlı</Badge>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {[
            { company: selectedSymbol, title: 'Q4 sonuçları beklentileri aştı', time: '2 dk önce', impact: 'Pozitif' },
            { company: selectedSymbol, title: 'Yönetim kurulu toplantısı duyurusu', time: '15 dk önce', impact: 'Nötr' },
            { company: selectedSymbol, title: 'Stratejik ortaklık anlaşması', time: '32 dk önce', impact: 'Pozitif' }
          ].map((item, index) => (
            <div key={index} className="p-3 bg-white/60 rounded border-l-2 border-green-400">
              <div className="flex items-center justify-between mb-1">
                <span className="font-medium text-sm">{item.company}</span>
                <Badge variant="outline" className="text-xs">{item.impact}</Badge>
              </div>
              <p className="text-sm text-slate-600">{item.title}</p>
              <p className="text-xs text-slate-500 mt-1">{item.time}</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
