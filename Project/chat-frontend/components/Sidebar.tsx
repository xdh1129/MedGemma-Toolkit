import React from 'react';
import { PlusCircle, MessageSquare, Settings, PanelLeftClose, Trash2 } from 'lucide-react';
import { ChatSession } from '../types';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  onNewChat: () => void;
  sessions: ChatSession[];
  currentSessionId: string | null;
  onSelectSession: (id: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ 
  isOpen, 
  onToggle, 
  onNewChat, 
  sessions, 
  currentSessionId, 
  onSelectSession 
}) => {
  return (
    <>
      {/* Mobile Overlay */}
      <div 
        className={`fixed inset-0 bg-black/30 z-20 md:hidden transition-opacity duration-300 ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
        onClick={onToggle}
      />

      {/* Sidebar Content */}
      <aside 
        className={`fixed md:relative z-30 h-full bg-slate-900 text-slate-300 w-64 transform transition-transform duration-300 ease-in-out flex flex-col ${isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0 md:w-0 md:overflow-hidden'}`}
      >
        <div className="p-4 flex items-center justify-between md:hidden">
            <h2 className="font-bold text-white">MedGemma</h2>
            <button onClick={onToggle} className="p-1 hover:text-white">
                <PanelLeftClose size={20} />
            </button>
        </div>

        <div className="p-3">
          <button 
            onClick={onNewChat}
            className="w-full flex items-center gap-3 px-4 py-3 bg-medical-600 hover:bg-medical-500 text-white rounded-lg transition-colors shadow-md group"
          >
            <PlusCircle size={20} />
            <span className="font-medium">New Chat</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-2">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 mt-2 px-2">
            History
          </div>
          
          {sessions.length === 0 ? (
            <div className="px-4 py-4 text-sm text-slate-600 text-center italic">
              No previous chats
            </div>
          ) : (
            <ul className="space-y-1">
              {sessions.map((session) => (
                <li key={session.id}>
                  <button 
                    onClick={() => onSelectSession(session.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors text-left text-sm 
                      ${currentSessionId === session.id 
                        ? 'bg-slate-800 text-white ring-1 ring-slate-700' 
                        : 'text-slate-300 hover:bg-slate-800/50 hover:text-white'
                      }`}
                  >
                    <MessageSquare size={16} className={`shrink-0 ${currentSessionId === session.id ? 'text-medical-400' : 'text-slate-500'}`} />
                    <span className="truncate">{session.title}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="p-4 border-t border-slate-800">
          <button className="flex items-center gap-3 text-sm text-slate-400 hover:text-white transition-colors w-full px-2 py-2 rounded-lg hover:bg-slate-800">
            <Settings size={18} />
            <span>Settings</span>
          </button>
          <div className="mt-4 flex items-center gap-3 px-2">
             <div className="w-8 h-8 rounded-full bg-gradient-to-br from-medical-400 to-blue-500 flex items-center justify-center text-white text-xs font-bold">
                MG
             </div>
             <div className="flex flex-col">
                 <span className="text-sm text-slate-200 font-medium">MedGemma Pro</span>
                 <span className="text-xs text-slate-500">v2.5 Flash</span>
             </div>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;