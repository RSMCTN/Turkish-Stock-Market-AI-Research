'use client';

import DailyDataImporter from '@/components/admin/DailyDataImporter';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Settings, Shield, Database, FileText } from 'lucide-react';

export default function AdminPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="container mx-auto px-4 py-8">
        
        {/* Admin Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">Yönetim Paneli</h1>
                <p className="text-gray-600">BIST Trading Dashboard Admin Center</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="bg-green-50 border-green-200 text-green-700">
                <Database className="h-3 w-3 mr-1" />
                Railway PostgreSQL
              </Badge>
              <Badge variant="outline" className="bg-blue-50 border-blue-200 text-blue-700">
                <Settings className="h-3 w-3 mr-1" />
                Admin Access
              </Badge>
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className="border-blue-200 bg-gradient-to-br from-blue-50 to-blue-100">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm text-blue-700">Toplam Hisse</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-800">589</div>
              <div className="text-xs text-blue-600">BIST aktif hisse senedi</div>
            </CardContent>
          </Card>
          
          <Card className="border-green-200 bg-gradient-to-br from-green-50 to-green-100">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm text-green-700">Son Güncelleme</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-800">Bugün</div>
              <div className="text-xs text-green-600">working_bist_data.json</div>
            </CardContent>
          </Card>
          
          <Card className="border-purple-200 bg-gradient-to-br from-purple-50 to-purple-100">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm text-purple-700">Database</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-800">Online</div>
              <div className="text-xs text-purple-600">Railway PostgreSQL</div>
            </CardContent>
          </Card>
          
          <Card className="border-orange-200 bg-gradient-to-br from-orange-50 to-orange-100">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm text-orange-700">API Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-orange-800">Active</div>
              <div className="text-xs text-orange-600">Backend endpoints</div>
            </CardContent>
          </Card>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card className="cursor-pointer hover:shadow-lg transition-shadow border-slate-200">
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <FileText className="h-4 w-4 text-blue-600" />
                Veri Aktarımı
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">Excel dosyalarından günlük BIST verilerini sisteme aktarın</p>
            </CardContent>
          </Card>
          
          <Card className="cursor-pointer hover:shadow-lg transition-shadow border-slate-200 opacity-50">
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <Database className="h-4 w-4 text-green-600" />
                Database Yönetimi
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">PostgreSQL veritabanı backup, restore ve bakım işlemleri</p>
              <Badge variant="outline" className="mt-2 text-xs">Yakında</Badge>
            </CardContent>
          </Card>
          
          <Card className="cursor-pointer hover:shadow-lg transition-shadow border-slate-200 opacity-50">
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <Settings className="h-4 w-4 text-purple-600" />
                Sistem Ayarları
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600">API endpoints, cache settings ve sistem konfigürasyonu</p>
              <Badge variant="outline" className="mt-2 text-xs">Yakında</Badge>
            </CardContent>
          </Card>
        </div>

        {/* Main Import Interface */}
        <DailyDataImporter />

        {/* Footer */}
        <div className="mt-12 text-center">
          <div className="flex items-center justify-center gap-2 text-sm text-gray-500">
            <Shield className="h-4 w-4" />
            <span>BIST Trading Dashboard Admin Panel</span>
            <span>•</span>
            <span>Güvenli Veri Yönetimi</span>
            <span>•</span>
            <span>© 2025</span>
          </div>
        </div>
      </div>
    </div>
  );
}
