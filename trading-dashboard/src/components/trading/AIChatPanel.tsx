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
  selectedSymbol = 'AKBNK',
  apiBaseUrl = 'https://bistai001-production.up.railway.app'
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
      // Call Railway production API
      const response = await fetch(`${apiBaseUrl}/api/ai-chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: inputText.trim(),
          symbol: selectedSymbol,
          context_type: contextType
        })
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const aiResponse = await response.json();

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
    <Card className="h-[600px] flex flex-col bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <CardHeader className="pb-3 border-b bg-white/80">
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

      <CardContent className="flex-1 flex flex-col p-4">
        {/* Messages Area */}
        <ScrollArea className="flex-1 mb-4 pr-2">
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
                  <div className={`inline-block p-3 rounded-lg max-w-[85%] ${
                    message.type === 'user'
                      ? 'bg-blue-500 text-white'
                      : 'bg-white border shadow-sm'
                  }`}>
                    <div className="whitespace-pre-wrap text-sm">
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

        {/* Sample Questions */}
        {messages.length <= 1 && (
          <div className="mb-4">
            <div className="text-xs text-gray-500 mb-2">💡 Örnek sorular:</div>
            <div className="grid grid-cols-1 gap-1">
              {getSampleQuestions().slice(0, 3).map((question, idx) => (
                <Button
                  key={idx}
                  variant="ghost"
                  size="sm"
                  onClick={() => setInputText(question)}
                  className="text-left text-xs h-auto p-2 justify-start text-gray-600 hover:text-blue-600"
                >
                  "{question}"
                </Button>
              ))}
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="flex gap-2">
          <Textarea
            placeholder={`${selectedSymbol} hakkında soru sorun...`}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
            className="flex-1 min-h-[40px] max-h-[100px] resize-none"
            rows={1}
          />
          <Button
            onClick={handleSendMessage}
            disabled={!inputText.trim() || isLoading}
            size="sm"
            className="px-3 bg-blue-600 hover:bg-blue-700"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>

        {/* Status Footer */}
        <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center gap-1">
            <Zap className="h-3 w-3" />
            Production Railway API
          </div>
          <div>
            Enter ile gönder • Shift+Enter yeni satır
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
