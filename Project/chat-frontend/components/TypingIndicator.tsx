import React from 'react';

interface TypingIndicatorProps {
  text?: string;
}

const TypingIndicator: React.FC<TypingIndicatorProps> = ({ text = '' }) => {
  return (
    <div className="flex items-center space-x-2 p-2 bg-white border border-slate-100 rounded-2xl shadow-sm ml-2">
      <div className="flex items-center space-x-1">
        <div className="w-2 h-2 bg-medical-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
        <div className="w-2 h-2 bg-medical-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
        <div className="w-2 h-2 bg-medical-400 rounded-full animate-bounce"></div>
      </div>
      {text && <span className="text-sm text-slate-500">{text}</span>}
    </div>
  );
};

export default TypingIndicator;