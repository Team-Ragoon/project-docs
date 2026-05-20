export type MessageRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
}

export interface ChatApiResponse {
  session_id: string;
  reply: string;
  pending_subject: string | null;
}
