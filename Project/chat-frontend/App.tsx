import React, { useState, useRef, useEffect, useCallback } from 'react';
import { SendHorizontal, Menu, AlertCircle, Paperclip, X } from 'lucide-react';

import Sidebar from './components/Sidebar';
import ChatBubble from './components/ChatBubble';
import { Message, Role, Attachment, ChatSession } from './types';
import { analyzeCaseStream } from './services/api';

const convertFileToDataUrl = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
  });
};

const App: React.FC = () => {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [selectedImage, setSelectedImage] = useState<Attachment | null>(null);

  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const startNewChat = useCallback(() => {
    setMessages([]);
    setCurrentSessionId(null);
    setSelectedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    setInput('');

    if (window.innerWidth < 768) {
      setIsSidebarOpen(false);
    }
  }, []);

  useEffect(() => {
    startNewChat();
  }, [startNewChat]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, selectedImage]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [input]);

  useEffect(() => {
    if (currentSessionId) {
      setSessions((prev) =>
        prev.map((session) =>
          session.id === currentSessionId ? { ...session, messages } : session,
        ),
      );
    }
  }, [messages, currentSessionId]);

  const loadSession = (sessionId: string) => {
    const session = sessions.find((s) => s.id === sessionId);
    if (!session) {
      return;
    }

    setCurrentSessionId(sessionId);
    setMessages(session.messages);
    setSelectedImage(null);
    setInput('');

    if (window.innerWidth < 768) {
      setIsSidebarOpen(false);
    }
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image.');
      return;
    }

    try {
      const dataUrl = await convertFileToDataUrl(file);
      setSelectedImage({
        mimeType: file.type,
        url: dataUrl,
        file,
      });
      setError(null);
    } catch (err) {
      console.error('File processing error', err);
      setError('Failed to process image.');
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSend = useCallback(async () => {
    const hasContent = input.trim().length > 0 || selectedImage !== null;
    if (!hasContent || isLoading) {
      return;
    }

    const userText = input.trim();
    const attachmentForMessage = selectedImage
      ? { mimeType: selectedImage.mimeType, url: selectedImage.url }
      : undefined;
    const imageFile = selectedImage?.file ?? null;

    setInput('');
    setSelectedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    const now = Date.now();
    const newUserMsg: Message = {
      id: now.toString(),
      role: Role.USER,
      content: userText,
      timestamp: now,
      attachment: attachmentForMessage,
    };
    const modelMsgId = `${now}-model`;
    const modelMsg: Message = {
      id: modelMsgId,
      role: Role.MODEL,
      content: '',
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, newUserMsg, modelMsg]);
    setIsLoading(true);
    setError(null);

    if (!currentSessionId) {
      const newSessionId = Date.now().toString();
      const titleText = userText.length > 0 ? userText : 'Image Query';
      const newSession: ChatSession = {
        id: newSessionId,
        title: titleText.slice(0, 30) + (titleText.length > 30 ? '...' : ''),
        date: Date.now(),
        messages: [newUserMsg],
      };

      setSessions((prev) => [newSession, ...prev]);
      setCurrentSessionId(newSessionId);
    }

    let vlmBuffer = '';
    let llmBuffer = '';

    const updateModelContent = (content: string) => {
      setMessages((prev) =>
        prev.map((msg) => (msg.id === modelMsgId ? { ...msg, content } : msg)),
      );
    };

    const refreshModelContent = () => {
      const formatted = llmBuffer.trim();
      updateModelContent(formatted);
    };

    try {
      await analyzeCaseStream(userText, imageFile, {
        onStatus: (state) => {
          if (state === 'queued') {
            updateModelContent('Queued... awaiting GPU availability.');
          } else if (state === 'processing') {
            updateModelContent('Analyzing...');
          }
        },
        onVlmToken: (token) => {
          vlmBuffer += token;
          refreshModelContent();
        },
        onLlmToken: (token) => {
          llmBuffer += token;
          refreshModelContent();
        },
        onDone: (payload) => {
          vlmBuffer = payload.vlm_output ?? vlmBuffer;
          llmBuffer = payload.llm_report ?? llmBuffer;
          refreshModelContent();
        },
        onError: (message) => {
          setError(message);
          updateModelContent(`Error: ${message}`);
        },
      });
    } catch (err) {
      console.error('Analysis error:', err);
      const detail =
        err instanceof Error ? err.message : 'Failed to get response. Please try again.';
      setError(detail);
      updateModelContent(`Error: ${detail}`);
    } finally {
      setIsLoading(false);
    }
  }, [input, selectedImage, isLoading, currentSessionId]);

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-screen bg-white overflow-hidden">
      <Sidebar
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        onNewChat={startNewChat}
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={loadSession}
      />

      <main className="flex-1 flex flex-col h-full relative min-w-0">
        <header className="flex items-center justify-between px-4 py-3 border-b border-slate-100 bg-white z-10">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="p-2 text-slate-500 hover:bg-slate-100 rounded-lg transition-colors"
            >
              <Menu size={20} />
            </button>
            <div>
              <h1 className="font-semibold text-slate-800">MedGemma</h1>
              <p className="text-xs text-medical-600 font-medium">Medical Assistant AI</p>
            </div>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto px-4 sm:px-6 md:px-8 py-6 scroll-smooth relative">
          <div className="max-w-3xl mx-auto min-h-full flex flex-col">
            {messages.length === 0 && (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-8 opacity-80 mt-10 sm:mt-0">
                <div className="w-16 h-16 bg-medical-50 rounded-2xl flex items-center justify-center mb-6 shadow-sm">
                  <svg
                    className="w-8 h-8 text-medical-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                    />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-slate-800 mb-2">Hello, I'm MedGemma.</h2>
                <p className="text-slate-500 max-w-md">
                  I can help answer questions about health conditions, medications, and general
                  wellness.
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-8 w-full max-w-xl">
                  <button
                    onClick={() => setInput('What are common symptoms of seasonal allergies?')}
                    className="p-4 border border-slate-200 rounded-xl text-left hover:bg-medical-50 hover:border-medical-200 transition-all text-sm text-slate-600"
                  >
                    Symptoms of allergies?
                  </button>
                  <button
                    onClick={() => setInput('Explain the difference between ibuprofen and acetaminophen.')}
                    className="p-4 border border-slate-200 rounded-xl text-left hover:bg-medical-50 hover:border-medical-200 transition-all text-sm text-slate-600"
                  >
                    Ibuprofen vs Acetaminophen
                  </button>
                </div>
              </div>
            )}

            {messages.map((msg) => (
              <ChatBubble key={msg.id} message={msg} />
            ))}

            {error && (
              <div className="mx-auto my-4 p-3 bg-red-50 text-red-600 text-sm rounded-lg flex items-center gap-2 border border-red-100">
                <AlertCircle size={16} />
                {error}
              </div>
            )}

            <div ref={messagesEndRef} className="h-4" />
          </div>
        </div>

        <div className="p-4 bg-white border-t border-slate-100">
          <div className="max-w-3xl mx-auto">
            {selectedImage && (
              <div className="flex items-center gap-2 mb-2 bg-slate-50 p-2 rounded-lg w-fit border border-slate-200">
                <img
                  src={selectedImage.url}
                  alt="Preview"
                  className="w-10 h-10 object-cover rounded-md"
                />
                <span className="text-xs text-slate-500 max-w-[100px] truncate">Image attached</span>
                <button
                  onClick={clearImage}
                  className="p-1 hover:bg-slate-200 rounded-full text-slate-500"
                >
                  <X size={14} />
                </button>
              </div>
            )}

            <div className="relative flex items-end gap-2 bg-slate-50 border border-slate-200 rounded-2xl p-2 focus-within:ring-2 focus-within:ring-medical-400/50 focus-within:border-medical-400 transition-all shadow-sm">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileSelect}
                accept="image/png, image/jpeg, image/webp, image/heic"
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-2 text-slate-400 hover:text-medical-600 hover:bg-slate-100 rounded-xl transition-colors mb-0.5"
                title="Attach image"
              >
                <Paperclip size={20} />
              </button>

              <textarea
                ref={textareaRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask MedGemma a question..."
                className="flex-1 bg-transparent border-none focus:ring-0 text-slate-800 placeholder:text-slate-400 resize-none max-h-40 py-2.5 px-1 scrollbar-hide"
                rows={1}
              />

              <button
                onClick={handleSend}
                disabled={isLoading || (!input.trim() && !selectedImage)}
                className="p-2 bg-medical-600 text-white rounded-xl hover:bg-medical-700 disabled:opacity-50 disabled:hover:bg-medical-600 transition-all shadow-sm mb-0.5"
              >
                <SendHorizontal size={18} />
              </button>
            </div>

            <div className="text-center mt-2">
              <p className="text-[10px] text-slate-400">
                MedGemma can make mistakes. Consider checking important medical information with a professional.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
