import axios from 'axios';
import { ChatApiResponse } from '../types';

const API_BASE = process.env.REACT_APP_API_URL ?? '';

const client = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
  timeout: 120000,
});

export const checkHealth = async (): Promise<boolean> => {
  try {
    await client.get('/api/health', { timeout: 5000 });
    return true;
  } catch {
    return false;
  }
};

export const sendChatMessage = async (
  message: string,
  sessionId: string | null
): Promise<ChatApiResponse> => {
  const { data } = await client.post<ChatApiResponse>('/api/chat', {
    message,
    session_id: sessionId,
  });
  return data;
};

export const resetSession = async (sessionId: string): Promise<void> => {
  await client.delete(`/api/session/${sessionId}`);
};
