import React from 'react';
import MessageList from './MessageList';
import { ChatMessage, ClarifyAnswer } from '../types';
import styles from '../styles/Chat.module.css';

interface ChatWindowProps {
  messages: ChatMessage[];
  loading: boolean;
  error: string | null;
  bottomRef: React.RefObject<HTMLDivElement>;
  onExampleClick: (example: string) => void;
  examplesDisabled?: boolean;
  showYesNoButtons?: boolean;
  pendingSubject?: string | null;
  onYesNoSelect?: (answer: ClarifyAnswer) => void;
  yesNoDisabled?: boolean;
}

const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  loading,
  error,
  bottomRef,
  onExampleClick,
  examplesDisabled,
  showYesNoButtons,
  pendingSubject,
  onYesNoSelect,
  yesNoDisabled,
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
        showYesNoButtons={showYesNoButtons}
        pendingSubject={pendingSubject}
        onYesNoSelect={onYesNoSelect}
        yesNoDisabled={yesNoDisabled}
      />
    </main>
  );
};

export default ChatWindow;
