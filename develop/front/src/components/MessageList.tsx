import React from 'react';
import MessageItem from './MessageItem';
import EmptyState from './EmptyState';
import { ChatMessage } from '../types';
import styles from '../styles/Chat.module.css';

interface MessageListProps {
  messages: ChatMessage[];
  loading: boolean;
  bottomRef: React.RefObject<HTMLDivElement>;
}

const MessageList: React.FC<MessageListProps> = ({ messages, loading, bottomRef }) => {
  return (
    <div className={styles.messageList}>
      {messages.length === 0 && !loading ? (
        <EmptyState />
      ) : (
        messages.map((message) => <MessageItem key={message.id} message={message} />)
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
