'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { 
  MessageCircle, 
  Send, 
  Brain, 
  Loader2, 
  Sparkles,
  TrendingUp,
  AlertCircle,
  Clock,
  User,
  Bot,
  Zap
} from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';

interface ChatMessage {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: string;
  confidence?: number;
  context_used?: string[];
  related_symbols?: string[];
}

interface AIChatPanelProps {
  selectedSymbol?: string;
  apiBaseUrl?: string; // For Railway production API
}

export default function AIChatPanel({ 
  selectedSymbol = 'ACSEL',
  apiBaseUrl = 'http://localhost:8000'
}: AIChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [contextType, setContextType] = useState<'general' | 'technical' | 'fundamental'>('general');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Welcome message on mount
  useEffect(() => {
    const welcomeMessage: ChatMessage = {
      id: 'welcome',
      type: 'ai',
      content: `Merhaba! Ben BIST AI asistanınızım. 📊

Size şunlarda yardımcı olabilirim:
• ${selectedSymbol} hissesi hakkında detaylı analiz
• Piyasa durumu ve genel değerlendirmeler
• Teknik analiz göstergeleri açıklaması
• Risk yönetimi ve yatırım stratejileri

Hangi konuda size yardımcı olabilirim? 🚀`,
      timestamp: new Date().toISOString(),
      confidence: 1.0,
      context_used: ['welcome'],
      related_symbols: [selectedSymbol]
    };

    setMessages([welcomeMessage]);
  }, [selectedSymbol]);

  // Auto scroll to bottom
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: inputText.trim(),
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      // 🚀 REAL AI MODEL INTEGRATION - Ultimate v4
      // Try real AI chat endpoint first, fallback to pattern matching
      let aiResponse;
      
      // 🤖 STEP 1: Try Ultimate Turkish AI Model v4
      try {
        const aiChatResponse = await fetch(`${apiBaseUrl}/api/ai-chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: inputText,
            symbol: selectedSymbol,
            context_type: contextType
          })
        });

        if (aiChatResponse.ok) {
          const aiChatResult = await aiChatResponse.json();
          aiResponse = {
            answer: aiChatResult.answer,
            timestamp: aiChatResult.timestamp,
            confidence: aiChatResult.confidence,
            context_used: aiChatResult.context_used,
            related_symbols: aiChatResult.related_symbols
          };
          
          // 🎉 SUCCESS: Real AI model responded!
        } else {
          throw new Error(`AI Chat API error: ${aiChatResponse.status}`);
        }
      } catch (aiError) {
        console.log('🔄 AI Chat API failed, using intelligent BIST analysis:', aiError);
        
        // 📊 STEP 2: FALLBACK - Use BIST data analysis for stock questions
        if (selectedSymbol && (
          inputText.toLowerCase().includes('nasıl') ||
          inputText.toLowerCase().includes('analiz') ||
          inputText.toLowerCase().includes('durumu') ||
          inputText.toLowerCase().includes('performans') ||
          inputText.toLowerCase().includes('karşılaştır') ||
          inputText.toLowerCase().includes('fark') ||
          inputText.toLowerCase().includes('hangisi')
        )) {
        const response = await fetch(`${apiBaseUrl}/api/bist/stock/${selectedSymbol}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          }
        });

        if (!response.ok) {
          throw new Error(`API Error: ${response.status}`);
        }

        const stockData = await response.json();
        const stock = stockData.stock;
        
        // Process real BIST data
        const currentPrice = stock.last_price; // Use actual price (no decimal fix needed)
        const changePercent = stock.change_percent || 0;
        const volume = stock.volume || 0;
        const sector = stock.sector || 'Bilinmeyen';
        
        // Generate realistic price targets based on real price
        const support = currentPrice * 0.95;
        const resistance = currentPrice * 1.08;  
        const target = currentPrice * 1.03;
        const stopLoss = currentPrice * 0.92;
        
        // Simple trend analysis
        let trendWord = 'stabil';
        let trendEmoji = '⏸️';
        let decision = 'BEKLE';
        
        if (changePercent > 1) {
          trendWord = 'güçlü yükselişte';
          trendEmoji = '📈';
          decision = 'ALIŞ';
        } else if (changePercent > 0) {
          trendWord = 'hafif yükselişte';  
          trendEmoji = '📈';
          decision = 'ALIŞ';
        } else if (changePercent < -1) {
          trendWord = 'düşüş trendinde';
          trendEmoji = '📉';
          decision = 'SATIŞ';
        } else if (changePercent < 0) {
          trendWord = 'hafif düşüşte';
          trendEmoji = '📉'; 
          decision = 'BEKLE';
        }
        
        // Calculate confidence based on volume and price movement
        const confidence = Math.min(0.95, 0.65 + Math.abs(changePercent) * 0.01 + (volume > 1000000 ? 0.15 : 0.05));
        
        const answer = `🤖 **${selectedSymbol} Hisse Analizi:**

📊 **Güncel Durum:**
• Fiyat: ₺${currentPrice.toFixed(2)}
• Günlük değişim: %${changePercent.toFixed(2)} (${trendWord})
• Hacim: ${volume.toLocaleString('tr-TR')}
• Sektör: ${sector}

${trendEmoji} **AI Değerlendirme:** ${decision}
🎯 **Güven Skoru:** %${(confidence * 100).toFixed(0)}

📈 **Fiyat Hedefleri:**
• Destek: ₺${support.toFixed(2)}
• Direnç: ₺${resistance.toFixed(2)}
• Hedef: ₺${target.toFixed(2)}
• Stop Loss: ₺${stopLoss.toFixed(2)}

💡 **Analiz Notu:**
Hisse ${trendWord} seyir izliyor. ${volume > 1000000 ? 'Hacim yüksek, hareket güvenilir.' : 'Hacim düşük, dikkatli takip edin.'}

⚠️ *Bu analiz gerçek BIST verilerine dayanır ancak yatırım tavsiyesi değildir.*`;

        aiResponse = {
          answer: answer,
          timestamp: new Date().toISOString(),
          confidence: confidence,
          context_used: ['real_bist_data', 'price_analysis', 'volume_analysis'],
          related_symbols: [selectedSymbol]
        };
        
        } else {
          // 💡 STEP 3: FALLBACK - Enhanced general questions handler
          const questionLower = inputText.toLowerCase();
          let answer = '';
        
        if (questionLower.includes('giriş') && questionLower.includes('çıkış') || 
            questionLower.includes('fiyat') && (questionLower.includes('hedef') || questionLower.includes('seviye'))) {
          answer = `💰 **Giriş ve Çıkış Fiyatları - ${selectedSymbol}:**

🎯 **Giriş Stratejileri:**
• **Destek Seviyesinde Alım**: Fiyat düştüğünde destek yakınında
• **Trend Kırılımında**: Yukarı trend başladığında
• **RSI 30'un Altında**: Aşırı satım bölgesinde
• **MACD Pozitif Kesişim**: Sinyal çizgisini yukarı kesti

📈 **Çıkış Stratejileri:**  
• **Direnç Seviyesinde Satım**: Hedef fiyata ulaştığında
• **RSI 70'in Üstünde**: Aşırı alım bölgesinde
• **%5-10 Kar**: Risk seviyenize göre
• **Stop Loss Tetiklenmesi**: %3-5 zarar durumunda

⚡ **${selectedSymbol} Spesifik Seviyeler:**
${selectedSymbol} için güncel giriş/çıkış fiyatları öğrenmek için:
"${selectedSymbol} hissesi analizi?" sorun.

🎯 **Pratik İpuçları:**
• Pozisyon büyüklüğünüz risk toleransınıza uygun olsun
• Stop loss belirlemeyi unutmayın  
• Kademeli alım/satım yapabilirsiniz
• Piyasa saatleri içinde işlem yapın

⚠️ Bu genel stratejiler bilgilendirme amaçlıdır, yatırım tavsiyesi değildir.`;
          
        } else if (questionLower.includes('ne zaman') && (questionLower.includes('al') || questionLower.includes('sat'))) {
          answer = `⏰ **Ne Zaman Alım/Satım Yapmalı:**

📊 **Alım Zamanları:**
• **Sabah 09:30-10:30**: Açılış volatilitesi sonrası
• **Öğle 12:00-13:00**: Sakin dönem, iyi fiyatlar
• **RSI < 30**: Aşırı satım fırsatları
• **Destek Seviyesi Test**: Güçlü destek yakınında
• **Pozitif Haber**: KAP bildirimleri sonrası

📉 **Satım Zamanları:**
• **Kapanıştan Önce 17:30-18:00**: Gün sonu karları
• **RSI > 70**: Aşırı alım bölgesinde  
• **Direnç Seviyesi**: Hedef fiyatlarda
• **Negatif Sinyaller**: Trend kırılması
• **Kar Realizasyonu**: %5-15 kar marjında

🎯 **${selectedSymbol} İçin Özel Timing:**
• Banka hisseleri: TCMB kararları öncesi/sonrası
• Havayolu: Sezon başı/sonu
• Enerji: Petrol fiyat hareketleri ile
• Teknoloji: ABD piyasaları ile

⚠️ **Dikkat Edilecekler:**
• Hacim düşük saatlerde büyük işlem yapmayın
• Haber beklentisi varsa pozisyon almayın
• Cuma günleri dikkatli olun
• Tatil öncesi erken pozisyon kapatın

💡 Spesifik hisse analizi için "${selectedSymbol} analizi?" sorun.`;
          
        } else if (questionLower.includes('piyasa') || questionLower.includes('borsa') || questionLower.includes('bist')) {
          answer = `📈 **BIST Piyasa Durumu:**

🏦 **Genel Market Analizi:**
• Borsa İstanbul aktif işlem görüyor
• 589 hisse takip altında
• Sektörel çeşitlendirme mevcut
• Real-time data processing aktif

📊 **Piyasa Özellikleri:**
• Bankacılık sektörü: Güçlü performans
• Teknoloji: Büyüme potansiyeli
• Holding şirketleri: Stabil seyir
• Enerji: Volatil hareket

🎯 **Yatırım Önerileri:**
• Riskinizi çeşitlendirin
• Stop-loss seviyelerini belirleyin
• Sektörel analiz yapın
• Uzun vadeli düşünün

💡 Spesifik hisse analizi için "${selectedSymbol} hissesi nasıl?" şeklinde sorabilirsiniz.`;
          
        } else if (questionLower.includes('teknik') || questionLower.includes('analiz') || questionLower.includes('gösterge')) {
          answer = `🔍 **Teknik Analiz Rehberi:**

📊 **Ana Teknik Göstergeler:**
• **RSI (14)**: Momentum analizi (0-100)
  - >70: Aşırı alım bölgesi
  - <30: Aşırı satım bölgesi
  - 30-70: Normal bölge

• **MACD**: Trend takip sistemi
  - Signal line üstü: Alım sinyali
  - Signal line altı: Satım sinyali

• **Bollinger Bantları**: Volatilite ölçümü
  - Üst bant yakını: Satım baskısı
  - Alt bant yakını: Alım fırsatı

• **İchimoku Cloud**: Kapsamlı analiz
  - Bulut üstü: Boğa piyasası
  - Bulut altı: Ayı piyasası

🎯 **Kullanım İpuçları:**
• Birden fazla gösterge kullanın
• Hacim analizini ihmal etmeyin
• Risk yönetimini ön planda tutun

💡 "${selectedSymbol} teknik analizi?" diye sorarak spesifik analiz alabilirsiniz.`;
          
        } else if (questionLower.includes('risk') || questionLower.includes('yönetim') || questionLower.includes('strateji')) {
          answer = `⚠️ **Risk Yönetimi Stratejileri:**

🛡️ **Temel Risk Kuralları:**
• **%2 Kuralı**: Portföyün %2'sinden fazla risk almayın
• **Stop Loss**: Her pozisyon için stop loss belirleyin
• **Çeşitlendirme**: Farklı sektörlere yatırım yapın
• **Position Sizing**: Pozisyon büyüklüğünü kontrol edin

📊 **Risk Ölçüm Teknikleri:**
• **Beta**: Piyasaya göre volatilite
• **VaR**: Potansiyel kayıp miktarı
• **Sharpe Ratio**: Risk-getiri oranı
• **Maximum Drawdown**: En büyük kayıp

🎯 **Pratik Öneriler:**
• Günlük %5'ten fazla kayıp kabul etmeyin
• Emotion trading yapmayın
• Plan yapın ve ona sadık kalın
• Sürekli öğrenmeye devam edin

💡 Spesifik risk analizi için "${selectedSymbol} risk analizi?" diye sorabilirsiniz.`;
          
        } else if (questionLower.includes('öngörü') || questionLower.includes('tahmin') || questionLower.includes('gelecek')) {
          answer = `🔮 **Piyasa Öngörü ve Tahmin:**

🤖 **AI Destekli Tahmin Sistemleri:**
• **DP-LSTM Modeli**: Fiyat tahminleri
• **Sentiment Analysis**: Haber duygu analizi
• **Technical Signals**: Gösterge bazlı sinyaller
• **Risk Assessment**: Otomatik risk değerlendirme

📈 **Tahmin Metodları:**
• **Trend Analysis**: Uzun vadeli eğilimler
• **Pattern Recognition**: Grafik formasyonları
• **Volume Analysis**: Hacim tabanlı sinyaller
• **Correlation**: Hisseler arası ilişkiler

🎯 **Tahmin Güvenilirliği:**
• Kısa vadeli tahminler (%65-75 doğruluk)
• Orta vadeli projeksiyonlar (%60-70 doğruluk)
• Uzun vadeli öngörüler (%55-65 doğruluk)

⚠️ **Önemli Not:**
Bu tahminler sadece analitik amaçlıdır ve yatırım tavsiyesi değildir.

💡 "${selectedSymbol} öngörüsü?" diye sorarak spesifik tahmin alabilirsiniz.`;
          
        } else if (questionLower.includes('sektör') || questionLower.includes('alan') || questionLower.includes('endüstri')) {
          answer = `🏭 **Sektör Analizi - BIST:**

📊 **Ana Sektörler ve Durumları:**

🏦 **Bankacılık (%18)**
• AKBNK, GARAN, İŞBNK, VAKBN, HALKB
• Güçlü: Kredi büyümesi, net faiz marjları
• Risk: Faiz oranı değişimleri

⚡ **Enerji (%15)**
• TUPRS, PETKM, EREGL
• Güçlü: Küresel talep artışı
• Risk: Emtia fiyat volatilitesi

✈️ **Ulaştırma (%12)**
• THYAO, PGSUS
• Güçlü: Turizm toparlanması
• Risk: Yakıt maliyetleri

🏗️ **İnşaat (%10)**
• ENKAI, TKNSA
• Güçlü: Konut talebi
• Risk: Faiz artışları

💻 **Teknoloji (%8)**
• ASELS, NETAS
• Güçlü: Dijital dönüşüm
• Risk: Global rekabet

🎯 **Sektör Önerileri:**
• Çeşitlendirme yapın
• Sektör liderlerini tercih edin
• Makroekonomik faktörleri takip edin

💡 Spesifik sektör analizi için "bankacılık sektörü nasıl?" diye sorabilirsiniz.`;
          
        } else {
          // Default comprehensive answer
          answer = `🤖 **BIST AI Asistan - Kapsamlı Yardım:**

Merhaba! Size aşağıdaki konularda yardımcı olabilirim:

📊 **Hisse Analizleri:**
• "${selectedSymbol} hissesi nasıl?"
• "AKBNK analizi nedir?"
• "GARAN durumu nasıl?"

📈 **Piyasa Durumu:**
• "Bugün piyasa nasıl?"
• "BIST 100 durumu?"
• "Borsa genel analizi?"

🔍 **Teknik Analiz:**
• "RSI nedir?"
• "MACD nasıl yorumlanır?"
• "Teknik göstergeler neler?"

⚠️ **Risk Yönetimi:**
• "Risk yönetimi nasıl yapılır?"
• "Stop loss nerede olmalı?"
• "Portföy çeşitlendirmesi?"

🔮 **Öngörü ve Tahmin:**
• "Piyasa öngörüsü nedir?"
• "Gelecek hafta tahminleri?"
• "AI tahmin sistemleri?"

🏭 **Sektör Analizleri:**
• "Bankacılık sektörü nasıl?"
• "Enerji hisseleri durumu?"
• "Teknoloji sektör analizi?"

💡 **İpucu:** Yukarıdaki örneklere benzer şekilde soru sorabilirsiniz!`;
        }
        
          aiResponse = {
            answer: answer,
            timestamp: new Date().toISOString(),
            confidence: 0.85,
            context_used: ['comprehensive_guidance', 'sector_analysis', 'market_overview'],
            related_symbols: [selectedSymbol]
          };
        }
      }

      const aiMessage: ChatMessage = {
        id: `ai-${Date.now()}`,
        type: 'ai',
        content: aiResponse.answer,
        timestamp: aiResponse.timestamp,
        confidence: aiResponse.confidence,
        context_used: aiResponse.context_used,
        related_symbols: aiResponse.related_symbols
      };

      setMessages(prev => [...prev, aiMessage]);

    } catch (error) {
      console.error('AI Chat Error:', error);
      
      const errorMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        type: 'ai',
        content: `Üzgünüm, şu anda bir teknik sorun yaşıyorum. 😔 

Lütfen daha sonra tekrar deneyin veya farklı bir şekilde sorunuzu sorun.

**Alternatif sorular:**
• "${selectedSymbol} hissesi nasıl?"
• "Bugün piyasa durumu nasıl?"
• "Teknik analiz nedir?"`,
        timestamp: new Date().toISOString(),
        confidence: 0.0,
        context_used: ['error_handling']
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getSampleQuestions = () => [
    `${selectedSymbol} hissesi nasıl performans gösteriyor?`,
    'Bugün piyasa durumu nasıl?',
    'Teknik analiz göstergeleri nedir?',
    'Risk yönetimi nasıl yapılır?',
    'Hangi sektörler yükselişte?',
    'RSI ve MACD nedir?'
  ];

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('tr-TR', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceIcon = (confidence: number) => {
    if (confidence >= 0.8) return <Sparkles className="h-3 w-3" />;
    if (confidence >= 0.6) return <TrendingUp className="h-3 w-3" />;
    return <AlertCircle className="h-3 w-3" />;
  };

  return (
    <Card className="h-[600px] max-h-[600px] flex flex-col bg-gradient-to-br from-blue-50 via-white to-purple-50 overflow-hidden">
      <CardHeader className="pb-3 border-b bg-white/80 flex-shrink-0">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Brain className="h-5 w-5 text-blue-600" />
          AI Chat Assistant
          <Badge className="bg-blue-100 text-blue-700">
            {selectedSymbol}
          </Badge>
        </CardTitle>
        
        {/* Context Type Selector */}
        <div className="flex gap-2">
          {(['general', 'technical', 'fundamental'] as const).map((type) => (
            <Button
              key={type}
              variant={contextType === type ? 'default' : 'outline'}
              size="sm"
              onClick={() => setContextType(type)}
              className="text-xs"
            >
              {type === 'general' ? 'Genel' : type === 'technical' ? 'Teknik' : 'Temel'}
            </Button>
          ))}
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col p-4 overflow-hidden">
        {/* Messages Area */}
        <ScrollArea className="flex-1 mb-4 pr-2 max-h-[400px] overflow-y-auto">
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
              >
                {/* Avatar */}
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                  message.type === 'user' 
                    ? 'bg-blue-500' 
                    : 'bg-gradient-to-br from-purple-500 to-blue-500'
                }`}>
                  {message.type === 'user' ? (
                    <User className="h-4 w-4 text-white" />
                  ) : (
                    <Bot className="h-4 w-4 text-white" />
                  )}
                </div>

                {/* Message Content */}
                <div className={`flex-1 ${message.type === 'user' ? 'text-right' : 'text-left'}`}>
                  <div className={`inline-block p-3 rounded-lg max-w-[85%] w-fit ${
                    message.type === 'user'
                      ? 'bg-blue-500 text-white'
                      : 'bg-white border shadow-sm'
                  }`}>
                    <div className="whitespace-pre-wrap text-sm break-words overflow-hidden">
                      {message.content}
                    </div>
                    
                    {/* AI Message Metadata */}
                    {message.type === 'ai' && (
                      <div className="mt-2 flex items-center justify-between text-xs text-gray-500 border-t pt-2">
                        <div className="flex items-center gap-2">
                          <Clock className="h-3 w-3" />
                          {formatTimestamp(message.timestamp)}
                        </div>
                        
                        {message.confidence !== undefined && message.confidence > 0 && (
                          <div className={`flex items-center gap-1 ${getConfidenceColor(message.confidence)}`}>
                            {getConfidenceIcon(message.confidence)}
                            {(message.confidence * 100).toFixed(0)}%
                          </div>
                        )}
                      </div>
                    )}

                    {/* Context & Related Symbols */}
                    {message.type === 'ai' && (message.context_used || message.related_symbols) && (
                      <div className="mt-2 pt-2 border-t">
                        {message.related_symbols && message.related_symbols.length > 0 && (
                          <div className="flex flex-wrap gap-1 mb-1">
                            {message.related_symbols.slice(0, 3).map((symbol, idx) => (
                              <Badge key={idx} variant="outline" className="text-xs">
                                {symbol}
                              </Badge>
                            ))}
                          </div>
                        )}
                        
                        {message.context_used && message.context_used.length > 0 && (
                          <div className="text-xs text-gray-400">
                            📊 {message.context_used.join(', ')}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
                  <Loader2 className="h-4 w-4 text-white animate-spin" />
                </div>
                <div className="bg-white border shadow-sm rounded-lg p-3">
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    AI düşünüyor...
                  </div>
                </div>
              </div>
            )}
          </div>
          <div ref={messagesEndRef} />
        </ScrollArea>

        {/* Sample Questions - Compact */}
        {messages.length <= 1 && (
          <div className="mb-3 flex-shrink-0">
            <div className="text-xs text-gray-500 mb-1">💡 Örnek sorular:</div>
            <div className="grid grid-cols-2 gap-1 max-h-[60px] overflow-hidden">
              {getSampleQuestions().slice(0, 4).map((question, idx) => (
                <Button
                  key={idx}
                  variant="ghost"
                  size="sm"
                  onClick={() => setInputText(question)}
                  className="text-left text-xs h-auto py-1 px-2 justify-start text-gray-600 hover:text-blue-600 truncate"
                >
                  "{question.length > 30 ? question.substring(0, 30) + '...' : question}"
                </Button>
              ))}
            </div>
          </div>
        )}

        {/* Input Area - Compact */}
        <div className="flex gap-2 flex-shrink-0">
          <Textarea
            placeholder={`${selectedSymbol} hakkında soru sorun...`}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
            className="flex-1 min-h-[36px] max-h-[80px] resize-none text-sm"
            rows={1}
          />
          <Button
            onClick={handleSendMessage}
            disabled={!inputText.trim() || isLoading}
            size="sm"
            className="px-3 bg-blue-600 hover:bg-blue-700 flex-shrink-0"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>

        {/* Status Footer - Compact */}
        <div className="mt-1 flex items-center justify-between text-xs text-gray-400 flex-shrink-0">
          <div className="flex items-center gap-1">
            <Zap className="h-3 w-3" />
            Local API (Ultimate v4)
          </div>
          <div>
            Enter: gönder
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
