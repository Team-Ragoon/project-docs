import React from 'react';
import MessageList from './MessageList';
import { ChatMessage } from '../types';
import styles from '../styles/Chat.module.css';

interface ChatWindowProps {
  messages: ChatMessage[];
  loading: boolean;
  error: string | null;
  bottomRef: React.RefObject<HTMLDivElement>;
  onExampleClick: (example: string) => void;
  examplesDisabled?: boolean;
}

const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  loading,
  error,
  bottomRef,
  onExampleClick,
  examplesDisabled,
}) => {
  return (
    <main className={styles.chatWindow}>
      {error && <div className={styles.errorBanner}>{error}</div>}
      <MessageList
        messages={messages}
        loading={loading}
        bottomRef={bottomRef}
        onExampleClick={onExampleClick}
        examplesDisabled={examplesDisabled}
      />
    </main>
  );
};

export default ChatWindow;
