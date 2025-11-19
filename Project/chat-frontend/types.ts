export enum Role {
  USER = 'user',
  MODEL = 'model'
}

export interface Attachment {
  mimeType: string;
  url: string;
  file?: File;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  timestamp: number;
  isStreaming?: boolean;
  attachment?: Attachment;
}

export interface ChatSession {
  id: string;
  title: string;
  date: number;
  messages: Message[];
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  currentInput: string;
}
