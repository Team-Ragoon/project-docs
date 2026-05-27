import React from 'react';
import MessageItem from './MessageItem';
import EmptyState from './EmptyState';
import YesNoButtons from './YesNoButtons';
import { ChatMessage, ClarifyAnswer } from '../types';
import styles from '../styles/Chat.module.css';

interface MessageListProps {
  messages: ChatMessage[];
  loading: boolean;
  bottomRef: React.RefObject<HTMLDivElement>;
  onExampleClick: (example: string) => void;
  examplesDisabled?: boolean;
  showYesNoButtons?: boolean;
  pendingSubject?: string | null;
  onYesNoSelect?: (answer: ClarifyAnswer) => void;
  yesNoDisabled?: boolean;
}

const MessageList: React.FC<MessageListProps> = ({
  messages,
  loading,
  bottomRef,
  onExampleClick,
  examplesDisabled,
  showYesNoButtons,
  pendingSubject,
  onYesNoSelect,
  yesNoDisabled,
}) => {
  const lastMessage = messages[messages.length - 1];
  const showButtons =
    showYesNoButtons &&
    !loading &&
    lastMessage?.role === 'assistant' &&
    onYesNoSelect;

  return (
    <div className={styles.messageList}>
      {messages.length === 0 && !loading ? (
        <EmptyState onExampleClick={onExampleClick} disabled={examplesDisabled} />
      ) : (
        messages.map((message) => <MessageItem key={message.id} message={message} />)
      )}
      {showButtons && (
        <YesNoButtons
          onSelect={onYesNoSelect}
          disabled={yesNoDisabled}
          pendingSubject={pendingSubject}
        />
      )}
      {loading && (
        <div className={styles.loadingRow}>
          <span className={styles.loadingDots}>답변 생성 중</span>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
};

export default MessageList;
