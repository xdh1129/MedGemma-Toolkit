import React from 'react';

const TypingIndicator: React.FC = () => {
  return (
    <div className="flex items-center space-x-1 p-2 bg-white border border-slate-100 rounded-2xl w-16 h-10 shadow-sm ml-2">
      <div className="w-2 h-2 bg-medical-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
      <div className="w-2 h-2 bg-medical-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
      <div className="w-2 h-2 bg-medical-400 rounded-full animate-bounce"></div>
    </div>
  );
};

export default TypingIndicator;