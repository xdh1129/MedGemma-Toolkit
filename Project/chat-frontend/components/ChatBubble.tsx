import React, { useState } from 'react';
import { Copy, ThumbsUp, ThumbsDown, Check, Bot, User } from 'lucide-react';
import { Message, Role } from '../types';
import MarkdownRenderer from './MarkdownRenderer';

interface ChatBubbleProps {
  message: Message;
}

const ChatBubble: React.FC<ChatBubbleProps> = ({ message }) => {
  const isUser = message.role === Role.USER;
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={`group flex w-full mb-6 ${
        isUser ? 'justify-end' : 'justify-start'
      } px-3 sm:px-6 lg:px-12 xl:px-20`}
    >
      <div
        className={`flex w-full max-w-3xl ${
          isUser ? 'flex-row-reverse' : 'flex-row'
        } gap-4 items-start`}
      >
        
        {/* Avatar */}
        <div className="flex-shrink-0">
          <div className={`w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center shadow-sm ${isUser ? 'bg-slate-200' : 'bg-medical-100 border border-medical-200'}`}>
            {isUser ? (
              <User size={20} className="text-slate-500" />
            ) : (
              <Bot size={24} className="text-medical-600" />
            )}
          </div>
        </div>

        {/* Message Content */}
        <div className={`flex flex-col min-w-0 ${isUser ? 'items-end' : 'items-start'}`}>
            
          {/* Name Label */}
          <span className="text-xs text-slate-400 mb-1 px-1">
            {isUser ? 'You' : 'MedGemma'}
          </span>

          <div 
            className={`relative px-5 py-4 rounded-2xl shadow-sm text-sm sm:text-base leading-relaxed overflow-hidden break-words w-full
              ${isUser 
                ? 'bg-white border border-slate-100 text-slate-800 rounded-tr-sm' 
                : 'bg-white border border-slate-100 text-slate-800 rounded-tl-sm'
              }`}
          >
             {/* Decorative Accent for Model */}
             {!isUser && <div className="absolute top-0 left-0 w-1 h-full bg-medical-500/50"></div>}
             
             {/* Image Attachment */}
             {message.attachment && (
               <div className="mb-3">
                 <img 
                   src={message.attachment.url} 
                   alt="User attachment" 
                   className="max-w-full rounded-lg border border-slate-200 max-h-64 object-cover"
                 />
               </div>
             )}

            <MarkdownRenderer content={message.content} />
          </div>

          {/* Action Buttons (Only for Model) */}
          {!isUser && !message.isStreaming && (
            <div className="flex items-center gap-2 mt-2 ml-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <button 
                onClick={handleCopy}
                className="p-1.5 text-slate-400 hover:text-medical-600 hover:bg-medical-50 rounded-md transition-colors"
                title="Copy to clipboard"
              >
                {copied ? <Check size={14} /> : <Copy size={14} />}
              </button>
              <button className="p-1.5 text-slate-400 hover:text-medical-600 hover:bg-medical-50 rounded-md transition-colors">
                <ThumbsUp size={14} />
              </button>
              <button className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-md transition-colors">
                <ThumbsDown size={14} />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatBubble;
