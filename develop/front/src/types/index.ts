export type MessageRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
}

export type AnswerMode = 'yes_no';
export type ClarifyAnswer = '예' | '아니요' | '모르겠음';

export interface ChatApiResponse {
  session_id: string;
  reply: string;
  pending_subject: string | null;
  answer_mode?: AnswerMode | null;
}
