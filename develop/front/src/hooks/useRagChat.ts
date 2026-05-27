import { useCallback, useEffect, useRef, useState } from 'react';
import { sendChatMessage, checkHealth } from '../services/api';
import { AnswerMode, ChatMessage } from '../types';

const createId = () =>
  `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

export const useRagChat = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pendingSubject, setPendingSubject] = useState<string | null>(null);
  const [answerMode, setAnswerMode] = useState<AnswerMode | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    let intervalId: ReturnType<typeof setInterval> | undefined;

    const runCheck = async () => {
      const ok = await checkHealth();
      if (cancelled) return;
      setApiOnline(ok);
      if (ok && intervalId !== undefined) {
        clearInterval(intervalId);
        intervalId = undefined;
      }
    };

    runCheck();
    intervalId = setInterval(runCheck, 5000);

    return () => {
      cancelled = true;
      if (intervalId !== undefined) clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || loading) return;

      const userMessage: ChatMessage = {
        id: createId(),
        role: 'user',
        content: trimmed,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setAnswerMode(null);
      setLoading(true);
      setError(null);

      try {
        const data = await sendChatMessage(trimmed, sessionId);
        setSessionId(data.session_id);
        setPendingSubject(data.pending_subject);
        setAnswerMode(data.answer_mode ?? null);

        const assistantMessage: ChatMessage = {
          id: createId(),
          role: 'assistant',
          content: data.reply,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : '답변을 가져오지 못했습니다. API 서버가 실행 중인지 확인하세요.';
        setError(message);
      } finally {
        setLoading(false);
      }
    },
    [loading, sessionId]
  );

  const startNewChat = useCallback(() => {
    setMessages([]);
    setSessionId(null);
    setPendingSubject(null);
    setAnswerMode(null);
    setError(null);
  }, []);

  return {
    messages,
    loading,
    error,
    pendingSubject,
    answerMode,
    apiOnline,
    sendMessage,
    startNewChat,
    bottomRef,
  };
};
